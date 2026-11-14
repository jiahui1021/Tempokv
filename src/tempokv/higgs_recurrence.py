"""Persistent prediction and HIGGS residual reconstruction for TempoKV."""

import triton
import triton.language as tl

from higgs_kernels.triton_ops import bshd_offsets, decode_group

from .higgs import _VECTOR_BITS, _grid
from .parameters import PreviousFrameParameters
from .runtime import persistent_launch_plan, persistent_units_per_launch


@triton.jit
def _reconstruct_kernel(
    codes, scales, grid, weights, bias, projection, output,
    code_b, code_s, scale_b, scale_s, proj_b, proj_s, proj_h, proj_d,
    out_b, out_s, out_h, out_d, sequence,
    heads: tl.constexpr, width: tl.constexpr, stride: tl.constexpr,
    group_size: tl.constexpr, stages: tl.constexpr, code_dim: tl.constexpr,
    bits: tl.constexpr, byte_count: tl.constexpr, half_transform: tl.constexpr,
    start_unit: tl.constexpr, chunk_units: tl.constexpr, history_length: tl.constexpr,
    per_head: tl.constexpr, previous_frame: tl.constexpr, cross_kv: tl.constexpr,
    legacy: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    group = tl.program_id(2)
    element = tl.arange(0, group_size)
    channel = group * group_size + element
    head = channel // width
    feature = channel % width
    valid_channel = channel < heads * width
    for unit in range(start_unit, start_unit + chunk_units):
        position = unit * stride + token
        if position < sequence:
            residual_position = position - stride
            residual = decode_group(
                codes + batch.to(tl.int64) * code_b + residual_position.to(tl.int64) * code_s + group * byte_count,
                scales + batch.to(tl.int64) * scale_b + residual_position.to(tl.int64) * scale_s + group,
                grid, element, True, group_size, stages, code_dim, bits, byte_count, half_transform,
            )
            prediction = tl.full((group_size,), 0, tl.float32)
            if cross_kv:
                projected = bshd_offsets(batch, position, channel, width, proj_b, proj_s, proj_h, proj_d)
                prediction = tl.load(projection + projected, valid_channel, 0).to(tl.float32)
            if history_length > 0:
                order = tl.minimum(unit, history_length)
                for lag in tl.static_range(history_length):
                    if lag < order:
                        prior = bshd_offsets(batch, position - (lag + 1) * stride, channel, width,
                                             out_b, out_s, out_h, out_d)
                        previous = tl.load(output + prior, valid_channel, 0).to(tl.float32)
                        if previous_frame:
                            prediction = previous
                        else:
                            if cross_kv:
                                offset = lag * heads * width + channel
                            elif per_head:
                                offset = ((order - 1) * history_length + lag) * heads * width + channel
                            else:
                                offset = ((order - 1) * history_length + lag) * width + feature
                            coefficient = tl.load(weights + offset, valid_channel, 0)
                            if legacy:
                                prediction += previous * coefficient
                            else:
                                prediction = tl.fma(previous, coefficient, prediction)
                if not cross_kv and not previous_frame:
                    if per_head:
                        bias_offset = (order - 1) * heads * width + head * width + feature
                    else:
                        bias_offset = (order - 1) * width + feature
                    prediction += tl.load(bias + bias_offset, valid_channel, 0)
            result_offset = bshd_offsets(batch, position, channel, width, out_b, out_s, out_h, out_d)
            tl.store(output + result_offset, prediction + residual, valid_channel)


def launch(packed, temporal, cross, projection, output):
    batch, sequence, heads, width = output.shape
    group_size = packed.residual_higgs_group
    vector = packed.bits in _VECTOR_BITS
    bits = 2 * packed.bits if vector else packed.bits
    dim = 2 if vector else 1
    byte_count = triton.cdiv(group_size // dim * bits, 8)
    groups = triton.cdiv(heads * width, group_size)
    grid, _ = _grid(packed.bits, output.device)
    codes = packed.residual_quant.squeeze(1)
    scales = packed.residual_scale.squeeze(1)
    if cross is not None:
        parameters = cross.to(output.device)
        weights, bias = parameters.history_weights, None
        projection = projection.transpose(1, 2)
    else:
        parameters = temporal.to(output.device)
        if isinstance(parameters, PreviousFrameParameters):
            weights, bias = None, None
        else:
            weights, bias = parameters.history_weights, parameters.history_bias
    history = parameters.history_length
    units = persistent_units_per_launch()
    if cross is not None or history > 4:
        units = min(units, 8)
    for shift_units, start, count in persistent_launch_plan(len(packed.unit_lengths), units, max(1, history)):
        shift = shift_units * packed.stride
        projected = None if projection is None else projection[:, shift:]
        _reconstruct_kernel[(batch, packed.stride, groups)](
            codes[:, shift:], scales[:, shift:], grid, weights, bias, projected, output[:, shift:],
            *codes.stride()[:2], *scales.stride()[:2],
            *(projected.stride() if projected is not None else (0, 0, 0, 0)),
            *output.stride(), sequence - shift, heads, width, packed.stride,
            group_size, group_size.bit_length() - 1, dim, bits, byte_count, vector,
            start, count, history, temporal is not None and temporal.per_head,
            isinstance(temporal, PreviousFrameParameters), cross is not None,
            packed.compatibility_mode == "legacy", num_warps=4, enable_fp_fusion=False,
        )
