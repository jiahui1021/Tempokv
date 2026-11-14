"""Bit-packed HIGGS kernels. Callers supply their grids and storage precision."""

import math

import torch
import triton
import triton.language as tl


def supports(group_size: int) -> bool:
    # Split/join require Triton 3.0. Keep the reference path available on 2.2.
    return 0 < group_size <= 4096 and not group_size & (group_size - 1) and hasattr(tl, "split")


@triton.jit
def _hadamard_stage(values, size: tl.constexpr, step: tl.constexpr):
    pairs = tl.reshape(values, (size // (2 * step), 2, step))
    left, right = tl.split(tl.permute(pairs, (0, 2, 1)))
    butterflies = tl.join(left + right, left - right)
    return tl.reshape(tl.permute(butterflies, (0, 2, 1)), (size,))


@triton.jit
def _hadamard(values, size: tl.constexpr, stages: tl.constexpr):
    for stage in tl.static_range(stages):
        values = _hadamard_stage(values, size, 1 << stage)
    return values


@triton.jit
def _vector_indices(rotated, grid, norms, count: tl.constexpr, dim: tl.constexpr, size: tl.constexpr):
    # Keep only a tile of centroid scores in registers. No scores are written to memory.
    tile: tl.constexpr = 16
    lane = tl.arange(0, tile)
    component = tl.reshape(rotated, (count, dim))
    if dim == 2:
        x, y = tl.split(component)
    else:
        even, odd = tl.split(tl.reshape(component, (count, 2, 2)))
        x, z = tl.split(even)
        y, w = tl.split(odd)
    best_score = tl.full((count,), -float("inf"), tl.float32)
    best_index = tl.full((count,), 0, tl.int32)
    for start in range(tl.cdiv(size, tile)):
        centroid = start * tile + lane
        base = grid + centroid * dim
        score = x[:, None] * tl.load(base, centroid < size, 0)[None, :]
        score = tl.fma(y[:, None], tl.load(base + 1, centroid < size, 0)[None, :], score)
        if dim == 4:
            score = tl.fma(z[:, None], tl.load(base + 2, centroid < size, 0)[None, :], score)
            score = tl.fma(w[:, None], tl.load(base + 3, centroid < size, 0)[None, :], score)
        norm = tl.load(norms + centroid, centroid < size, 0)
        score = tl.where(centroid[None, :] < size, score - norm[None, :], -float("inf"))
        maximum, index = tl.max(score, axis=1, return_indices=True, return_indices_tie_break_left=True)
        best_index = tl.where(maximum > best_score, start * tile + index, best_index)
        best_score = tl.maximum(maximum, best_score)
    return best_index


@triton.jit
def _scalar_indices(rotated, boundaries, bits: tl.constexpr, size: tl.constexpr):
    index = tl.full((size,), 0, tl.int32)
    # Match bucketize(right=False) by advancing only past a strictly lower boundary.
    for step in tl.static_range(bits - 1, -1, -1):
        candidate = index + (1 << step)
        boundary = tl.load(boundaries + candidate - 1)
        index = tl.where(rotated > boundary, candidate, index)
    return tl.where(rotated == rotated, index, (1 << bits) - 1)


@triton.jit
def _pad_right(values, size: tl.constexpr, padded_size: tl.constexpr):
    repeat = tl.arange(0, padded_size // size)
    return tl.reshape(tl.where(repeat[:, None] == 0, values[None, :], 0), (padded_size,))


@triton.jit
def _pack_group(
    output, indices, bits: tl.constexpr, count: tl.constexpr,
    codes_per_word: tl.constexpr, bytes_per_word: tl.constexpr,
    byte_lanes: tl.constexpr, word_codes: tl.constexpr, byte_count: tl.constexpr,
):
    if count < codes_per_word:
        indices = _pad_right(indices, count, codes_per_word)
    if codes_per_word * bits > 32:
        indices = indices.to(tl.uint64)
    else:
        indices = indices.to(tl.uint32)
    if codes_per_word == 1:
        tl.store(output + tl.arange(0, count), indices)
    else:
        lane = tl.arange(0, word_codes) % codes_per_word
        shifts = (codes_per_word - 1 - lane) * bits
        words = tl.sum(
            tl.reshape(indices << shifts, (word_codes // codes_per_word, codes_per_word)), axis=1
        )
        byte = tl.arange(0, byte_lanes)
        position = tl.arange(0, word_codes // codes_per_word)[:, None] * bytes_per_word + byte[None, :]
        shift = tl.maximum(bytes_per_word - 1 - byte, 0) * 8
        tl.store(output + position, (words[:, None] >> shift[None, :]).to(tl.uint8),
                 (byte[None, :] < bytes_per_word) & (position < byte_count))


@triton.jit
def _quantize_kernel(
    value, grid, search, codes, scales, stride_b, stride_s, stride_h, stride_d,
    tokens, heads: tl.constexpr, width: tl.constexpr, groups: tl.constexpr,
    group_size: tl.constexpr, stages: tl.constexpr, code_dim: tl.constexpr,
    bits: tl.constexpr, code_count: tl.constexpr, byte_count: tl.constexpr,
    codes_per_word: tl.constexpr, bytes_per_word: tl.constexpr, byte_lanes: tl.constexpr,
    word_codes: tl.constexpr, zero_code: tl.constexpr, check_finite: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    flat = group * group_size + tl.arange(0, group_size)
    offset = (
        (row // tokens).to(tl.int64) * stride_b + (row % tokens).to(tl.int64) * stride_s
        + (flat // width) * stride_h + (flat % width) * stride_d
    )
    values = tl.load(value + offset, flat < heads * width, 0).to(tl.float32)
    scale = tl.sqrt(tl.sum(values * values))
    if check_finite:
        tl.device_assert(scale < float("inf"), "non-finite HIGGS scales")
    rotated = tl.div_rn(_hadamard(values, group_size, stages), tl.where(scale == 0, 1.0, scale))
    if code_dim == 1:
        indices = _scalar_indices(rotated, search, bits, group_size)
    else:
        if code_dim > group_size:
            rotated = _pad_right(rotated, group_size, code_dim)
        indices = _vector_indices(rotated * 2, grid, search, code_count, code_dim, 1 << bits)
        if zero_code:
            indices = tl.where(scale == 0, 0, indices)
    group_offset = row.to(tl.int64) * groups + group
    _pack_group(
        codes + group_offset * byte_count, indices, bits, code_count,
        codes_per_word, bytes_per_word, byte_lanes, word_codes, byte_count,
    )
    tl.store(scales + group_offset, scale)


@triton.jit
def decode_group(
    codes, scales, grid, element, valid,
    group_size: tl.constexpr, stages: tl.constexpr, code_dim: tl.constexpr,
    bits: tl.constexpr, byte_count: tl.constexpr, half_transform: tl.constexpr,
):
    if bits == 8:
        index = tl.load(codes + element // code_dim, valid, 0).to(tl.int32)
    else:
        bit = (element // code_dim) * bits
        byte = bit // 8
        first = tl.load(codes + byte, valid, 0).to(tl.int32)
        if 8 % bits == 0:
            index = (first >> (8 - bit % 8 - bits)) & ((1 << bits) - 1)
        else:
            second = tl.load(codes + byte + 1, valid & (byte + 1 < byte_count), 0).to(tl.int32)
            index = (((first << 8) | second) >> (16 - bit % 8 - bits)) & ((1 << bits) - 1)
    scale = tl.load(scales, valid, 0).to(tl.float32)
    rotated = tl.load(grid + index * code_dim + element % code_dim).to(tl.float32) * scale
    if half_transform:
        rotated = rotated.to(tl.float16).to(tl.float32)
    restored = _hadamard(rotated, group_size, stages) * (1.0 / group_size)
    if half_transform:
        restored = restored.to(tl.float16).to(tl.float32)
    return restored


@triton.jit
def bshd_offsets(batch, token, channel, width: tl.constexpr, stride_b, stride_s, stride_h, stride_d):
    return (
        batch.to(tl.int64) * stride_b + token.to(tl.int64) * stride_s
        + (channel // width) * stride_h + (channel % width) * stride_d
    )


@triton.jit
def _dequantize_kernel(
    codes, scales, grid, predictor, output,
    code_b, code_s, scale_b, scale_s, pred_b, pred_s, pred_h, pred_d,
    out_b, out_s, out_h, out_d, tokens, heads: tl.constexpr, width: tl.constexpr,
    group_size: tl.constexpr, stages: tl.constexpr, code_dim: tl.constexpr,
    bits: tl.constexpr, byte_count: tl.constexpr, half_transform: tl.constexpr,
    add_predictor: tl.constexpr, residual_dtype: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    batch, token = row // tokens, row % tokens
    element = tl.arange(0, group_size)
    channel = group * group_size + element
    valid = channel < heads * width
    restored = decode_group(
        codes + batch.to(tl.int64) * code_b + token.to(tl.int64) * code_s + group * byte_count,
        scales + batch.to(tl.int64) * scale_b + token.to(tl.int64) * scale_s + group,
        grid, element, True, group_size, stages, code_dim, bits, byte_count, half_transform,
    )
    if add_predictor:
        pred_offset = bshd_offsets(batch, token, channel, width, pred_b, pred_s, pred_h, pred_d)
        prediction = tl.load(predictor + pred_offset, valid, 0).to(tl.float32)
        restored = prediction + restored.to(residual_dtype).to(tl.float32)
    offset = bshd_offsets(batch, token, channel, width, out_b, out_s, out_h, out_d)
    tl.store(output + offset, restored, valid)


def quantize(
    value, grid, search, *, group_size, code_bits, code_dim, zero_code=False, check_finite=False,
):
    """Return packed [B, S, groups, bytes] codes and [B, S, groups] scales."""
    batch, tokens, heads, width = value.shape
    groups = triton.cdiv(heads * width, group_size)
    count = triton.cdiv(group_size, code_dim)
    byte_count = triton.cdiv(count * code_bits, 8)
    codes = torch.empty((batch, tokens, groups, byte_count), device=value.device, dtype=torch.uint8)
    scales = torch.empty((batch, tokens, groups), device=value.device, dtype=torch.float32)
    if batch and tokens:
        gcd = math.gcd(8, code_bits)
        codes_per_word, bytes_per_word = 8 // gcd, code_bits // gcd
        _quantize_kernel[(batch * tokens, groups)](
            value, grid, search, codes, scales, *value.stride(), tokens, heads, width, groups,
            group_size, group_size.bit_length() - 1, code_dim, code_bits, count, byte_count,
            codes_per_word, bytes_per_word, triton.next_power_of_2(bytes_per_word),
            max(count, codes_per_word), zero_code, check_finite,
            num_warps=8 if code_dim > 1 and group_size >= 512 else 4, enable_fp_fusion=False,
            debug=check_finite,
        )
    return codes, scales


_TRITON_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def dequantize_into(
    codes, scales, grid, output, *, group_size, code_bits, code_dim, half_transform,
    predictor=None, residual_dtype=torch.float32,
):
    """Decode into [B, S, H, D], optionally adding a predictor before each store."""
    batch, tokens, heads, width = output.shape
    if not batch or not tokens:
        return
    groups = triton.cdiv(heads * width, group_size)
    byte_count = triton.cdiv(triton.cdiv(group_size, code_dim) * code_bits, 8)
    _dequantize_kernel[(batch * tokens, groups)](
        codes, scales, grid, predictor, output, *codes.stride()[:2], *scales.stride()[:2],
        *(predictor.stride() if predictor is not None else (0, 0, 0, 0)),
        *output.stride(), tokens, heads, width, group_size, group_size.bit_length() - 1,
        code_dim, code_bits, byte_count, half_transform, predictor is not None,
        _TRITON_DTYPES[residual_dtype], num_warps=4, enable_fp_fusion=False,
    )
