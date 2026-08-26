from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .runtime import persistent_units_per_launch

if TYPE_CHECKING:
    from .codec import PackedTensor


_ROUTE_COUNTS: dict[str, int] = {}


def route_report() -> dict[str, int]:
    return dict(_ROUTE_COUNTS)


def reset_route_report() -> None:
    _ROUTE_COUNTS.clear()


def _record(route: str) -> None:
    _ROUTE_COUNTS[route] = _ROUTE_COUNTS.get(route, 0) + 1


def _decline(reason: str) -> None:
    """Name why a span fell back, so a slow run says which gate rejected it.

    A silent ``return None`` is indistinguishable from a kernel that never
    existed: the caller only sees the Torch decoder running, or, under strict
    Triton, a bare "not supported" with nothing to act on.
    """
    _record(f"declined:{reason}")


def _route_name(route: str, direct_output_layout: str | None) -> str:
    if direct_output_layout is None:
        return route
    # Preserve the established BSHD route name used by the Self-Forcing
    # benchmark contract.  BHSD is named explicitly so reports still
    # distinguish the two direct-output layouts.
    if direct_output_layout == "BSHD":
        return f"{route}_direct"
    return f"{route}_direct_bhsd"


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
    sequence,
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
def _direct_persistent_decode_kernel(
    quant_ptr,
    scale_ptr,
    zero_ptr,
    output_ptr,
    output_stride_b,
    output_stride_h,
    output_stride_s,
    output_stride_d,
    sequence,
    residual_sequence,
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
):
    """Decode several direct-quantized units in one launch."""
    batch_head = tl.program_id(0)
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
            value = (unsigned.to(tl.float32) - zero) * scale
        else:
            value = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale
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
        tl.store(
            output_ptr + output_offset,
            value.to(tl.bfloat16).to(tl.float32),
            mask=valid,
        )


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
    sequence,
    residual_sequence,
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
            bias = tl.load(bias_ptr + bias_offset, mask=channel < width, other=0.0)
            prediction = bias[None, :] + tl.zeros(
                (block_s, block_d), tl.float32
            )
        else:
            bias_offset = (order - 1) * width + channel
            bias = tl.load(bias_ptr + bias_offset, mask=channel < width, other=0.0)
            prediction = bias[None, :] + tl.zeros(
                (block_s, block_d), tl.float32
            )
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
                        (((order - 1) * history_length + lag) * heads + head)
                        * width
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
    sequence,
    residual_sequence,
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
    cross_input_precision: tl.constexpr,
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
    # Cross(K)'s projection matrix is invariant across every unit owned by
    # this program.  Load the tile once instead of issuing the same load from
    # each unrolled iteration.
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
        prediction = tl.dot(
            source,
            weight,
            out_dtype=tl.float32,
            input_precision=cross_input_precision,
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
    bias_ptr,
    history_weight_ptr,
    output_ptr,
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
    asymmetric: tl.constexpr,
    cross_input_precision: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
    block_k: tl.constexpr,
):
    """Stream Cross(K) and the V recurrence without a span-sized FP32 buffer.

    The projection is issued as a ``tl.dot`` over the tile this program owns
    and consumed on the spot.  Precomputing it into a tensor instead reads and
    writes a buffer the size of the whole V span once each, and that traffic
    costs far more than the projection arithmetic it was meant to amortize.

    Only diagonal history reaches this kernel: a per-lag dense matrix mixes
    output channels, which would force one program to own all of them and
    collapse the parallelism the ``block_d`` tiling exists to provide.
    """
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
    # The Cross(K) matrix tile is shared by all recurrent units in this
    # program.  Keeping the load outside the unrolled loop avoids redundant
    # memory instructions while preserving the FP32 projection exactly.
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
    # V history length one is the production route.  Seed its recurrence from
    # the unit immediately before this launch and then carry the BF16-rounded
    # value in registers.  The explicit round preserves the old store/load
    # semantics while avoiding a dependent global-memory read on every unit.
    if history_length == 1:
        previous_coefficient = tl.load(
            history_weight_ptr + head * width + channel,
            mask=channel < width,
            other=0.0,
        ).to(tl.float32)
        previous_token = (start_unit - 1) * stride + token
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
        previous_unit = tl.load(
            output_ptr + previous_offset,
            mask=(token[:, None] < stride)
            & (previous_token[:, None] < sequence)
            & (channel[None, :] < width),
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
        prediction = tl.dot(
            source,
            weight,
            out_dtype=tl.float32,
            input_precision=cross_input_precision,
        ) + bias[None, :]

        if history_length == 1:
            prediction += previous_unit * previous_coefficient[None, :]
        else:
            for lag in tl.static_range(0, history_length):
                if unit_index > lag:
                    coefficient = tl.load(
                        history_weight_ptr
                        + (lag * heads + head) * width
                        + channel,
                        mask=channel < width,
                        other=0.0,
                    ).to(tl.float32)
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
                        output_ptr + previous_offset,
                        mask=(token[:, None] < stride)
                        & (previous_token[:, None] < sequence)
                        & (channel[None, :] < width),
                        other=0.0,
                    ).to(tl.float32)
                    prediction += previous * coefficient[None, :]
        current = (prediction + residual).to(tl.bfloat16).to(tl.float32)
        if history_length == 1:
            previous_unit = current
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
    output_layout: str,
) -> tuple[torch.Tensor, str | None] | None:
    batch, heads, sequence, width = packed.shape
    device = packed.anchor_quant.device
    if output_layout not in {"BHSD", "BSHD"}:
        _decline("output_layout")
        return None
    if output_tensor is None:
        return (
            torch.empty(packed.shape, device=device, dtype=output_dtype),
            None,
        )
    expected_shape = (
        (batch, heads, sequence, width)
        if output_layout == "BHSD"
        else (batch, sequence, heads, width)
    )
    dimensions = sorted(
        (stride, size)
        for size, stride in zip(output_tensor.shape, output_tensor.stride())
        if size > 1
    )
    covered_span = 1
    safe_strides = all(stride > 0 for stride in output_tensor.stride())
    for stride, size in dimensions:
        if stride < covered_span:
            safe_strides = False
            break
        covered_span += (size - 1) * stride
    if (
        not output_tensor.is_cuda
        or output_tensor.device != device
        or output_tensor.dtype != output_dtype
        or tuple(output_tensor.shape) != expected_shape
        or not safe_strides
    ):
        _decline("output_tensor_layout")
        return None
    return output_tensor, output_layout


def _logical_bhsd_output(
    output: torch.Tensor,
    direct_output_layout: str | None,
) -> torch.Tensor:
    return (
        output.permute(0, 2, 1, 3)
        if direct_output_layout == "BSHD"
        else output
    )


def decode(
    packed: "PackedTensor",
    *,
    source: torch.Tensor | None,
    output_dtype: torch.dtype,
    output_tensor: torch.Tensor | None = None,
    output_layout: str = "BSHD",
) -> torch.Tensor | None:
    """Fused GPU prediction and residual decode for TempoKV's two runtime routes."""
    if not _supported(packed, source):
        _decline("unsupported_state")
        return None
    batch, heads, sequence, width = packed.shape
    batch_heads = batch * heads
    block_s = 16
    block_d = max(16, min(128, triton.next_power_of_2(width)))
    scale_width = width // packed.group_size
    prepared_output = _prepare_output(
        packed, output_dtype, output_tensor, output_layout
    )
    if prepared_output is None:
        return None
    output, direct_output_layout = prepared_output
    if direct_output_layout == "BSHD":
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
        _record(_route_name("anchor", direct_output_layout))
        return _logical_bhsd_output(output, direct_output_layout)
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

    if packed.predictor_kind == "temporal":
        assert packed.temporal is not None
        parameters = packed.temporal.to(output.device)
        if parameters.history_length == 0:
            if output_dtype != torch.bfloat16:
                _decline("direct_output_dtype")
                return None
            if not _uniform_layout(packed):
                _decline("direct_unit_lengths")
                return None
            grid = (
                batch_heads,
                _ceil_div(packed.stride, block_s),
                _ceil_div(width, block_d),
            )
            persistent_units = persistent_units_per_launch()
            for start_unit in range(1, len(packed.unit_lengths), persistent_units):
                chunk_units = min(
                    persistent_units, len(packed.unit_lengths) - start_unit
                )
                _direct_persistent_decode_kernel[grid](
                    quant,
                    scale,
                    zero,
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
                    start_unit,
                    chunk_units,
                    packed.bits,
                    packed.group_size,
                    asymmetric,
                    block_s,
                    block_d,
                )
            _record(_route_name("direct_persistent", direct_output_layout))
            return _logical_bhsd_output(output, direct_output_layout)
        # Only the persistent route remains.  A span that misses this gate is
        # handed back to the Torch decoder by name instead of falling to a
        # second, per-unit kernel that production never reached.
        if output_dtype != torch.bfloat16:
            _decline("temporal_output_dtype")
            return None
        if parameters.history_length > 8:
            _decline("temporal_history_length")
            return None
        if not _uniform_layout(packed):
            _decline("temporal_unit_lengths")
            return None
        grid = (
            batch_heads,
            _ceil_div(packed.stride, block_s),
            _ceil_div(width, block_d),
        )
        persistent_units = persistent_units_per_launch()
        # Large K orders create a nested compile-time expansion.  Keep the
        # original bound unless the production-size order is small.
        if parameters.history_length > 4:
            persistent_units = min(persistent_units, 8)
        for start_unit in range(1, len(packed.unit_lengths), persistent_units):
            chunk_units = min(
                persistent_units, len(packed.unit_lengths) - start_unit
            )
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
        _record(_route_name("temporal_persistent", direct_output_layout))
        return _logical_bhsd_output(output, direct_output_layout)

    assert source is not None and packed.cross_kv is not None
    parameters = packed.cross_kv.to(output.device)
    if parameters.layered:
        _decline("layered_parameters")
        return None
    if parameters.history_length and not parameters.history_diagonal:
        # A dense per-lag matrix mixes output channels, so one program would
        # have to own all of them and the block_d tiling would buy nothing.
        # The Torch decoder applies every lag.
        _decline("dense_v_history")
        return None
    source_stride_b, source_stride_h, source_stride_s, source_stride_d = (
        source.stride()
    )
    # CrossKVParameters normalizes every tensor to contiguous storage when it
    # is constructed (including per-layer selections).  Avoid repeating the
    # contiguous dispatch on every cache read.
    weight = parameters.weight
    history_weight = (
        parameters.history_weights
        if parameters.history_length
        else weight
    )
    bias = parameters.bias
    block_k = max(16, triton.next_power_of_2(width))
    cross_block_d = 32 if width >= 128 else block_d
    # As with the temporal route, only the persistent kernels remain, so a span
    # that misses the gate names the reason and goes to the Torch decoder.
    if output_dtype != torch.bfloat16 or source.dtype != torch.bfloat16:
        _decline("cross_kv_dtype")
        return None
    if width != 128:
        _decline("cross_kv_head_dim")
        return None
    if not _uniform_layout(packed):
        _decline("cross_kv_unit_lengths")
        return None
    grid = (
        batch_heads,
        _ceil_div(packed.stride, block_s),
        _ceil_div(width, cross_block_d),
    )
    # Both Cross routes unroll a K projection per unit, so a launch batch
    # holds one source and one weight tile per unrolled unit.  Cap it where
    # the reference implementation does rather than at the temporal route's
    # larger default, which only has to keep elementwise state alive.
    persistent_units = min(persistent_units_per_launch(), 8)
    for start_unit in range(
        1, len(packed.unit_lengths), persistent_units
    ):
        chunk_units = min(
            persistent_units, len(packed.unit_lengths) - start_unit
        )
        if parameters.history_length:
            _cross_kv_history_persistent_decode_kernel[grid](
                quant,
                scale,
                zero,
                source,
                weight,
                bias,
                history_weight,
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
                start_unit,
                chunk_units,
                packed.bits,
                packed.group_size,
                asymmetric,
                packed.cross_matmul_precision,
                block_s,
                cross_block_d,
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
                packed.cross_matmul_precision,
                block_s,
                cross_block_d,
                block_k,
            )
    _record(_route_name("cross_kv_persistent", direct_output_layout))
    return _logical_bhsd_output(output, direct_output_layout)
