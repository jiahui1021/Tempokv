from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .quantization import MAX_BITS
from .parameters import PreviousFrameParameters
from .runtime import persistent_launch_plan, persistent_units_per_launch

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
    """Record why a span fell back to the Torch decoder."""
    _record(f"declined:{reason}")


def _route_name(route: str, direct_output_layout: str | None) -> str:
    if direct_output_layout is None:
        return route
    # BSHD keeps the historical route name; BHSD is suffixed.
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
def _load_packed_unsigned(
    quant_ptr,
    row_offset,
    channel,
    valid,
    packed_width: tl.constexpr,
    bits: tl.constexpr,
):
    """Read one MSB-first packed code, including codes crossing byte boundaries."""
    if bits == 8:
        unsigned = tl.load(
            quant_ptr + row_offset + channel[None, :], mask=valid, other=0
        ).to(tl.int32)
    elif (bits == 1 or bits == 2) or bits == 4:
        lanes = 8 // bits
        packed_channel = channel // lanes
        lane = channel - packed_channel * lanes
        shift = bits * (lanes - lane - 1)
        byte = tl.load(
            quant_ptr + row_offset + packed_channel[None, :],
            mask=valid,
            other=0,
        ).to(tl.int32)
        unsigned = (byte >> shift[None, :]) & ((1 << bits) - 1)
    else:
        bit_position = channel * bits
        packed_channel = bit_position // 8
        bit_offset = bit_position - packed_channel * 8
        first = tl.load(
            quant_ptr + row_offset + packed_channel[None, :],
            mask=valid & (packed_channel[None, :] < packed_width),
            other=0,
        ).to(tl.int32)
        second = tl.load(
            quant_ptr + row_offset + packed_channel[None, :] + 1,
            mask=valid & (packed_channel[None, :] + 1 < packed_width),
            other=0,
        ).to(tl.int32)
        # Must match ``_window_bytes`` in quantization.py: 2 bytes up to INT8, 3 above.
        if bits <= 8:
            word = (first << 8) | second
            shift = 16 - bit_offset - bits
        else:
            third = tl.load(
                quant_ptr + row_offset + packed_channel[None, :] + 2,
                mask=valid & (packed_channel[None, :] + 2 < packed_width),
                other=0,
            ).to(tl.int32)
            word = (first << 16) | (second << 8) | third
            shift = 24 - bit_offset - bits
        unsigned = (word >> shift[None, :]) & ((1 << bits) - 1)
    return unsigned


@triton.jit
def _dequantize_symmetric_code(unsigned, scale, bits: tl.constexpr):
    if bits == 1:
        value = (unsigned.to(tl.float32) * 2.0 - 1.0) * scale
    else:
        value = (unsigned - ((1 << (bits - 1)) - 1)).to(tl.float32) * scale
    return value


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
    quant_row = (
        batch_head * unit_length * packed_width
        + token[:, None] * packed_width
    )
    unsigned = _load_packed_unsigned(
        quant_ptr, quant_row, channel, valid, packed_width, bits
    )
    group = channel // group_size
    scale_offset = (
        batch_head * unit_length * scale_width
        + token[:, None] * scale_width
        + group[None, :]
    )
    scale = tl.load(scale_ptr + scale_offset, mask=valid, other=1.0).to(tl.float32)
    value = _dequantize_symmetric_code(unsigned, scale, bits)
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
        quant_row = (
            batch_head * residual_sequence * packed_width
            + residual_token[:, None] * packed_width
        )
        unsigned = _load_packed_unsigned(
            quant_ptr, quant_row, channel, valid, packed_width, bits
        )
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
            value = _dequantize_symmetric_code(unsigned, scale, bits)
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
    previous_frame: tl.constexpr = False,
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
        quant_row = (
            batch_head * residual_sequence * packed_width
            + residual_token[:, None] * packed_width
        )
        unsigned = _load_packed_unsigned(
            quant_ptr, quant_row, channel, valid, packed_width, bits
        )
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
            residual = _dequantize_symmetric_code(unsigned, scale, bits)

        if previous_frame:
            previous_offset = _strided_bhsd_offsets(
                batch_head, (unit_index - 1) * stride + token, channel,
                heads, output_stride_b, output_stride_h, output_stride_s, output_stride_d,
            )
            prediction = tl.load(output_ptr + previous_offset, mask=valid, other=0.0).to(tl.float32)
        else:
            order = unit_index
            if order > history_length:
                order = history_length
            if per_head:
                bias_offset = ((order - 1) * heads + head) * width + channel
                bias = tl.load(bias_ptr + bias_offset, mask=channel < width, other=0.0)
                prediction = bias[None, :] + tl.zeros((block_s, block_d), tl.float32)
            else:
                bias_offset = (order - 1) * width + channel
                bias = tl.load(bias_ptr + bias_offset, mask=channel < width, other=0.0)
                prediction = bias[None, :] + tl.zeros((block_s, block_d), tl.float32)
            for lag in tl.static_range(0, history_length):
                if lag < order:
                    previous_token = (unit_index - lag - 1) * stride + token
                    previous_offset = _strided_bhsd_offsets(
                        batch_head, previous_token, channel, heads,
                        output_stride_b, output_stride_h, output_stride_s, output_stride_d,
                    )
                    previous = tl.load(output_ptr + previous_offset, mask=valid, other=0.0).to(tl.float32)
                    if per_head:
                        weight_offset = (((order - 1) * history_length + lag) * heads + head) * width + channel
                    else:
                        weight_offset = ((order - 1) * history_length + lag) * width + channel
                    coefficient = tl.load(weight_ptr + weight_offset, mask=channel < width, other=0.0)
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
    # Load the Cross(K) tile once per program.
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
        quant_row = (
            batch_head * residual_sequence * packed_width
            + residual_token[:, None] * packed_width
        )
        unsigned = _load_packed_unsigned(
            quant_ptr, quant_row, channel, valid, packed_width, bits
        )
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
            residual = _dequantize_symmetric_code(unsigned, scale, bits)
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

    The projection is a per-tile ``tl.dot`` consumed on the spot; only diagonal
    V history is supported, since dense history would mix output channels.
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
    # Load the Cross(K) tile once per program.
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
    # History length 1 (production): carry the BF16-rounded previous unit in
    # registers instead of reloading it.
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
        quant_row = (
            batch_head * residual_sequence * packed_width
            + residual_token[:, None] * packed_width
        )
        unsigned = _load_packed_unsigned(
            quant_ptr, quant_row, channel, valid, packed_width, bits
        )
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
            residual = _dequantize_symmetric_code(unsigned, scale, bits)

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
    # Residuals stop at 8 bits; anchors cover INT1-INT15 (raw anchors are copied).
    if not 1 <= packed.bits <= 8 or not 1 <= packed.anchor_bits <= MAX_BITS:
        return False
    if packed.predictor_kind in {"cross_kv", "innovation"}:
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


def _advance(tensor: torch.Tensor, dim: int, offset: int) -> torch.Tensor:
    """View of ``tensor`` starting ``offset`` along ``dim``; strides unchanged."""
    if not offset:
        return tensor
    return tensor.narrow(dim, offset, tensor.shape[dim] - offset)


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
        output_sequence_dim = 1
    else:
        output_stride_b, output_stride_h, output_stride_s, output_stride_d = (
            output.stride()
        )
        output_sequence_dim = 2
    anchor_length = packed.unit_lengths[0]
    if packed.raw_anchor:
        # Raw anchor: copy through the logical BHSD view; the slice cast converts dtype.
        _logical_bhsd_output(output, direct_output_layout)[
            :, :, :anchor_length
        ] = packed.anchor_quant
    else:
        anchor_packed_width = (width * packed.anchor_bits + 7) // 8
        anchor_grid = (
            batch_heads,
            _ceil_div(anchor_length, block_s),
            _ceil_div(width, block_d),
        )
        _anchor_decode_kernel[anchor_grid](
            packed.anchor_quant.contiguous().view(
                batch_heads, anchor_length, anchor_packed_width
            ),
            packed.anchor_scale.contiguous().view(
                batch_heads, anchor_length, scale_width
            ),
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
        # Name the copy for what it is.  Reporting it as the anchor kernel would
        # put time in the profile against a launch that never happened.
        _record(
            _route_name(
                "anchor_raw" if packed.raw_anchor else "anchor",
                direct_output_layout,
            )
        )
        return _logical_bhsd_output(output, direct_output_layout)
    assert packed.residual_scale is not None
    residual_sequence = int(packed.residual_quant.shape[2])
    residual_packed_width = (width * packed.bits + 7) // 8
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
            for shift_units, start_unit, chunk_units in persistent_launch_plan(
                len(packed.unit_lengths), persistent_units_per_launch(), 1
            ):
                shift = shift_units * packed.stride
                _direct_persistent_decode_kernel[grid](
                    _advance(quant, 1, shift),
                    _advance(scale, 1, shift),
                    _advance(zero, 1, shift),
                    _advance(output, output_sequence_dim, shift),
                    output_stride_b,
                    output_stride_h,
                    output_stride_s,
                    output_stride_d,
                    sequence - shift,
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
        # Only the persistent route exists; anything else falls back to Torch by name.
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
        # From unit ``history_length`` on, a unit reads its full history, so
        # later launches share one start_unit instead of compiling per offset.
        for shift_units, start_unit, chunk_units in persistent_launch_plan(
            len(packed.unit_lengths),
            persistent_units,
            max(1, parameters.history_length),
        ):
            shift = shift_units * packed.stride
            _temporal_persistent_decode_kernel[grid](
                _advance(quant, 1, shift),
                _advance(scale, 1, shift),
                _advance(zero, 1, shift),
                None if isinstance(parameters, PreviousFrameParameters) else parameters.history_weights.contiguous(),
                None if isinstance(parameters, PreviousFrameParameters) else parameters.history_bias.contiguous(),
                _advance(output, output_sequence_dim, shift),
                output_stride_b,
                output_stride_h,
                output_stride_s,
                output_stride_d,
                sequence - shift,
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
                isinstance(parameters, PreviousFrameParameters),
            )
        _record(_route_name(
            "previous_frame_persistent" if isinstance(parameters, PreviousFrameParameters) else "temporal_persistent",
            direct_output_layout,
        ))
        return _logical_bhsd_output(output, direct_output_layout)

    assert source is not None and packed.cross_kv is not None
    parameters = packed.cross_kv.to(output.device)
    if parameters.layered:
        _decline("layered_parameters")
        return None
    if parameters.history_length and not parameters.history_diagonal:
        # Dense per-lag history mixes channels; the Torch decoder handles it.
        _decline("dense_v_history")
        return None
    source_stride_b, source_stride_h, source_stride_s, source_stride_d = (
        source.stride()
    )
    # CrossKVParameters tensors are already contiguous.
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
    # Cross routes keep a source and weight tile per unrolled unit, so cap the batch.
    persistent_units = min(persistent_units_per_launch(), 8)
    for shift_units, start_unit, chunk_units in persistent_launch_plan(
        len(packed.unit_lengths),
        persistent_units,
        max(1, parameters.history_length),
    ):
        shift = shift_units * packed.stride
        launch_quant = _advance(quant, 1, shift)
        launch_scale = _advance(scale, 1, shift)
        launch_zero = _advance(zero, 1, shift)
        launch_source = _advance(source, 2, shift)
        launch_output = _advance(output, output_sequence_dim, shift)
        if parameters.history_length:
            _cross_kv_history_persistent_decode_kernel[grid](
                launch_quant,
                launch_scale,
                launch_zero,
                launch_source,
                weight,
                bias,
                history_weight,
                launch_output,
                source_stride_b,
                source_stride_h,
                source_stride_s,
                source_stride_d,
                output_stride_b,
                output_stride_h,
                output_stride_s,
                output_stride_d,
                sequence - shift,
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
                launch_quant,
                launch_scale,
                launch_zero,
                launch_source,
                weight,
                bias,
                launch_output,
                source_stride_b,
                source_stride_h,
                source_stride_s,
                source_stride_d,
                output_stride_b,
                output_stride_h,
                output_stride_s,
                output_stride_d,
                sequence - shift,
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
    route = (
        "innovation_persistent"
        if packed.predictor_kind == "innovation"
        else "cross_kv_persistent"
    )
    _record(_route_name(route, direct_output_layout))
    return _logical_bhsd_output(output, direct_output_layout)
