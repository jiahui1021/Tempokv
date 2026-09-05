from __future__ import annotations

import torch


# The packer's own ceiling.  Residual widths stop at 8 by configuration
# because a wider residual costs more than the bfloat16 value it replaces, and
# quantized anchors stop at 15 because at 16 the codes plus their group scale
# also cost more than bfloat16 -- see RAW_ANCHOR_BITS in config.py.  Nothing
# currently asks for 16, and the packer is written for the width rather than
# for its callers, so the bound stays where the layout stops working.
MAX_BITS = 16


def _validate(value: torch.Tensor, bits: int, group_size: int) -> None:
    _validate_bits(bits)
    if value.ndim != 4:
        raise ValueError(f"expected [B,H,S,D], got {tuple(value.shape)}")
    if group_size <= 0 or value.shape[-1] % group_size:
        raise ValueError("group_size must divide the head dimension")


def _validate_bits(bits: int) -> None:
    if not 1 <= bits <= MAX_BITS:
        raise ValueError(f"bits must be between 1 and {MAX_BITS}")


def code_dtype(bits: int) -> torch.dtype:
    """Smallest integer dtype that holds one unsigned ``bits``-wide code."""
    return torch.uint8 if bits <= 8 else torch.int32


def _window_bytes(bits: int) -> int:
    """Bytes one code can touch: its own width plus a 7-bit byte offset.

    A wider window is harmless — the extra bytes receive zeros — so one size
    per range is enough, and the Triton decoder can mirror the same rule
    instead of recomputing a per-width span.
    """
    return 2 if bits <= 8 else 3


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

    window = _window_bytes(bits)
    window_bits = 8 * window
    bit_start = torch.arange(width, device=values.device, dtype=torch.int64) * bits
    byte_index = torch.div(bit_start, 8, rounding_mode="floor")
    bit_offset = bit_start.remainder(8)
    # Left-align each code inside its window, then cut the window into bytes.
    aligned = torch.bitwise_left_shift(codes, window_bits - bit_offset - bits)

    # The scratch columns absorb the window bytes that reach past the payload.
    # Every code ends inside the payload, so those bytes only ever receive
    # zeros and the slice below discards nothing.  Distinct codes never share a
    # bit, so accumulating is the same as OR-ing them together.
    packed = torch.zeros(
        (rows, packed_width + window - 1), dtype=torch.int64, device=values.device
    )
    for position in range(window):
        byte = torch.bitwise_and(
            torch.bitwise_right_shift(aligned, window_bits - 8 * (position + 1)),
            0xFF,
        )
        packed.scatter_add_(1, (byte_index + position).expand(rows, -1), byte)
    return (
        packed[:, :packed_width]
        .to(torch.uint8)
        .reshape(*values.shape[:-1], packed_width)
    )


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
    span = _window_bytes(bits)
    # The zero bytes make the window valid for a code ending in the final
    # byte, without a special case for non-crossing codes.
    padded = torch.nn.functional.pad(packed_rows, (0, span - 1))
    bit_start = torch.arange(width, device=packed.device, dtype=torch.int64) * bits
    byte_index = torch.div(bit_start, 8, rounding_mode="floor")
    bit_offset = bit_start.remainder(8)
    window = padded[:, byte_index]
    for position in range(1, span):
        window = torch.bitwise_or(
            torch.bitwise_left_shift(window, 8), padded[:, byte_index + position]
        )
    shift = 8 * span - bit_offset - bits
    codes = torch.bitwise_and(
        torch.bitwise_right_shift(window, shift), (1 << bits) - 1
    )
    return codes.to(code_dtype(bits)).reshape(*packed.shape[:-1], width)


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
    unsigned = (quant + qmax).to(code_dtype(bits)).reshape_as(value)
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
    packed = _pack(quant.to(code_dtype(bits)).reshape_as(value), bits)
    return packed, scale.squeeze(-1), zero_point.squeeze(-1).to(code_dtype(bits))


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
