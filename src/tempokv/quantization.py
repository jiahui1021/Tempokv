from __future__ import annotations

import torch


def _validate(value: torch.Tensor, bits: int, group_size: int) -> None:
    if bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4, or 8")
    if value.ndim != 4:
        raise ValueError(f"expected [B,H,S,D], got {tuple(value.shape)}")
    if group_size <= 0 or value.shape[-1] % group_size:
        raise ValueError("group_size must divide the head dimension")
    lanes = 8 // bits
    if value.shape[-1] % lanes:
        raise ValueError(
            f"head dimension must be divisible by {lanes} for INT{bits} packing"
        )


def _pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return values.to(torch.uint8)
    lanes = 8 // bits
    shape = (*values.shape[:-1], values.shape[-1] // lanes, lanes)
    grouped = values.reshape(shape).to(torch.uint8)
    packed = torch.zeros(shape[:-1], dtype=torch.uint8, device=values.device)
    for lane in range(lanes):
        packed |= grouped[..., lane] << (bits * (lanes - lane - 1))
    return packed


def _unpack(packed: torch.Tensor, bits: int, width: int) -> torch.Tensor:
    if bits == 8:
        return packed.to(torch.uint8)
    lanes = 8 // bits
    if width % lanes:
        raise ValueError("head dimension is not packable at this bit width")
    parts = [
        (packed >> (bits * (lanes - lane - 1))) & ((1 << bits) - 1)
        for lane in range(lanes)
    ]
    return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], width)


def quantize_symmetric(
    value: torch.Tensor,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(value, bits, group_size)
    groups = value.float().reshape(*value.shape[:-1], -1, group_size)
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
