from __future__ import annotations

import torch


def _validate(value: torch.Tensor, bits: int, group_size: int) -> None:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    if value.ndim != 4:
        raise ValueError(f"expected [B,H,S,D], got {tuple(value.shape)}")
    if group_size <= 0 or value.shape[-1] % group_size:
        raise ValueError("group_size must divide the head dimension")


def _validate_bits(bits: int) -> None:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")


def _pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the final dimension into a contiguous MSB-first bit stream.

    Code zero begins at the most-significant bit of byte zero.  A code may
    cross a byte boundary, and unused low bits in the final byte are zero.
    This is byte-for-byte compatible with the previous INT2/INT4/INT8 layout.
    """
    _validate_bits(bits)
    width = int(values.shape[-1])
    if width <= 0:
        raise ValueError("width must be positive")
    packed_width = (width * bits + 7) // 8
    if 8 % bits == 0:
        if bits == 8:
            return values.to(torch.uint8)
        # Preserve the original, allocation-light layout and hot path for
        # power-of-two bit widths.  Padding is confined to unused low bits of
        # the final byte, so widths no longer need to be lane-aligned.
        lanes = 8 // bits
        padding = (-width) % lanes
        codes = values.to(torch.uint8)
        if padding:
            codes = torch.nn.functional.pad(codes, (0, padding))
        grouped = codes.reshape(*values.shape[:-1], packed_width, lanes)
        packed = torch.zeros(
            grouped.shape[:-1], dtype=torch.uint8, device=values.device
        )
        for lane in range(lanes):
            packed |= grouped[..., lane] << (bits * (lanes - lane - 1))
        return packed

    rows = values.numel() // width
    codes = values.reshape(rows, width).to(torch.int64)

    bit_start = torch.arange(width, device=values.device, dtype=torch.int64) * bits
    byte_index = torch.div(bit_start, 8, rounding_mode="floor")
    bit_offset = bit_start.remainder(8)
    first_bits = torch.minimum(8 - bit_offset, torch.full_like(bit_offset, bits))
    first = torch.bitwise_right_shift(codes, bits - first_bits)
    first = torch.bitwise_left_shift(first, 8 - bit_offset - first_bits)

    packed = torch.zeros(
        (rows, packed_width), dtype=torch.int64, device=values.device
    )
    packed.scatter_add_(1, byte_index.expand(rows, -1), first)

    crosses = first_bits < bits
    remaining = bits - first_bits[crosses]
    tail_mask = torch.bitwise_left_shift(torch.ones_like(remaining), remaining) - 1
    tail = torch.bitwise_and(codes[:, crosses], tail_mask)
    tail = torch.bitwise_left_shift(tail, 8 - remaining)
    packed.scatter_add_(
        1,
        (byte_index[crosses] + 1).expand(rows, -1),
        tail,
    )
    return packed.to(torch.uint8).reshape(*values.shape[:-1], packed_width)


def _unpack(packed: torch.Tensor, bits: int, width: int) -> torch.Tensor:
    """Reverse :func:`_pack` for an explicitly supplied logical width."""
    _validate_bits(bits)
    if width <= 0:
        raise ValueError("width must be positive")
    expected_width = (width * bits + 7) // 8
    if packed.shape[-1] != expected_width:
        raise ValueError(
            f"packed width must be {expected_width} bytes for {width} INT{bits} values"
        )
    if 8 % bits == 0:
        if bits == 8:
            return packed.to(torch.uint8)
        lanes = 8 // bits
        parts = [
            (packed >> (bits * (lanes - lane - 1))) & ((1 << bits) - 1)
            for lane in range(lanes)
        ]
        unpacked = torch.stack(parts, dim=-1).reshape(
            *packed.shape[:-1], expected_width * lanes
        )
        return unpacked[..., :width]

    rows = packed.numel() // expected_width
    packed_rows = packed.reshape(rows, expected_width).to(torch.int64)
    # The zero byte makes the two-byte window valid for a code ending in the
    # final byte, without a special case for non-crossing codes.
    padded = torch.nn.functional.pad(packed_rows, (0, 1))
    bit_start = torch.arange(width, device=packed.device, dtype=torch.int64) * bits
    byte_index = torch.div(bit_start, 8, rounding_mode="floor")
    bit_offset = bit_start.remainder(8)
    window = torch.bitwise_left_shift(padded[:, byte_index], 8)
    window = torch.bitwise_or(window, padded[:, byte_index + 1])
    shift = 16 - bit_offset - bits
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(window, shift), (1 << bits) - 1
    )
    return codes.to(torch.uint8).reshape(*packed.shape[:-1], width)


def quantize_symmetric(
    value: torch.Tensor,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(value, bits, group_size)
    groups = value.float().reshape(*value.shape[:-1], -1, group_size)
    if bits == 1:
        # Binary symmetric quantization has no exact zero code.  Use a stable
        # sign convention: zero maps to +scale, while code 0/1 represents
        # -scale/+scale respectively.
        scale = _checked_scale(
            groups.abs().amax(dim=-1).clamp_min(1e-12), scale_dtype
        )
        unsigned = (groups >= 0).to(torch.uint8).reshape_as(value)
        return _pack(unsigned, bits), scale
    qmax = (1 << (bits - 1)) - 1
    # Round the scale to its stored precision *before* quantizing against it.
    # The decoder only ever sees the stored scale, so quantizing against a
    # higher-precision one leaves a gain error the residual never absorbs.
    scale = _checked_scale(
        groups.abs().amax(dim=-1).clamp_min(1e-12).div(qmax), scale_dtype
    )
    quant = torch.round(groups / scale.float().unsqueeze(-1)).clamp(-qmax, qmax)
    unsigned = (quant + qmax).to(torch.uint8).reshape_as(value)
    return _pack(unsigned, bits), scale


# float16 tops out at 65504, so it is the only supported scale dtype a finite
# float32 group can overflow; float32 and bfloat16 share float32's exponent
# range.  Restricting the check to float16 keeps the production bfloat16 path
# free of an extra reduction on the quantization hot path.
_NARROW_SCALE_DTYPES = (torch.float16,)


def _checked_scale(scale: "torch.Tensor", scale_dtype: "torch.dtype") -> "torch.Tensor":
    """Round a scale to its stored dtype, refusing one that does not survive.

    Storing the scale in float16 silently overflows to ``inf`` once the group
    magnitude is large enough, and every value dequantized against it comes
    back NaN.  Failing here names the cause; the alternative is a run that
    produces NaN video and no explanation.
    """
    stored = scale.to(scale_dtype)
    if scale_dtype in _NARROW_SCALE_DTYPES and not torch.isfinite(stored).all():
        finite = scale[torch.isfinite(scale)]
        largest = float(finite.max()) if finite.numel() else float("inf")
        raise ValueError(
            f"quantization scale overflows {scale_dtype}: the largest finite "
            f"scale is {largest:g}. Use a wider scale_dtype, a smaller "
            "group_size, or rescale the input."
        )
    return stored


def dequantize_symmetric(
    packed: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    group_size: int,
    width: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    unsigned = _unpack(packed, bits, width)
    if bits == 1:
        quant = unsigned.to(torch.float32).mul(2).sub(1)
        groups = quant.reshape(*quant.shape[:-1], -1, group_size)
        return (groups * scale.float().unsqueeze(-1)).reshape(*quant.shape).to(
            output_dtype
        )
    qmax = (1 << (bits - 1)) - 1
    quant = unsigned.to(torch.float32).sub(qmax)
    groups = quant.reshape(*quant.shape[:-1], -1, group_size)
    return (groups * scale.float().unsqueeze(-1)).reshape(*quant.shape).to(output_dtype)


def quantize_asymmetric(
    value: torch.Tensor,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate(value, bits, group_size)
    groups = value.float().reshape(*value.shape[:-1], -1, group_size)
    zeros = torch.zeros_like(groups[..., :1])
    low = torch.minimum(groups.amin(dim=-1, keepdim=True), zeros)
    high = torch.maximum(groups.amax(dim=-1, keepdim=True), zeros)
    qmax = (1 << bits) - 1
    # Same ordering as the symmetric path: the zero point and the codes must be
    # derived from the scale the decoder will actually use.
    scale = _checked_scale((high - low).clamp_min(1e-12).div(qmax), scale_dtype)
    zero_point = torch.round(-low / scale.float()).clamp(0, qmax)
    quant = torch.round(groups / scale.float() + zero_point).clamp(0, qmax)
    packed = _pack(quant.to(torch.uint8).reshape_as(value), bits)
    return packed, scale.squeeze(-1), zero_point.squeeze(-1).to(torch.uint8)


def dequantize_asymmetric(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bits: int,
    group_size: int,
    width: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    quant = _unpack(packed, bits, width).to(torch.float32)
    groups = quant.reshape(*quant.shape[:-1], -1, group_size)
    result = (groups - zero_point.float().unsqueeze(-1)) * scale.float().unsqueeze(-1)
    return result.reshape(*quant.shape).to(output_dtype)


def roundtrip(
    value: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
    symmetric: bool,
) -> torch.Tensor:
    """Quantize then dequantize an arbitrary ``[..., H, D]`` slice.

    The quantizers take packed ``[B,H,S,D]`` caches; trace and rollout code
    works on single frames, so reshape around them here instead of in every
    caller.
    """
    original_shape = value.shape
    head_count, dim = map(int, original_shape[-2:])
    packed = value.float().reshape(-1, head_count, 1, dim)
    if symmetric:
        quant, scale = quantize_symmetric(packed, bits, group_size, scale_dtype)
        rebuilt = dequantize_symmetric(
            quant, scale, bits, group_size, dim, torch.float32
        )
    else:
        quant, scale, zero_point = quantize_asymmetric(
            packed, bits, group_size, scale_dtype
        )
        rebuilt = dequantize_asymmetric(
            quant, scale, zero_point, bits, group_size, dim, torch.float32
        )
    return rebuilt.reshape(original_shape)
