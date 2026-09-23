"""One-pass float64 read-time RoPE for a pre-RoPE K cache read.

The reference in :mod:`tempokv.runtime.rope` rotates in complex128 to match
upstream ``causal_rope_apply`` bit for bit, but it does so with separate
PyTorch kernels per model block: widen to float64, multiply, narrow to BF16,
copy into the result.  Every one of those passes streams the whole float64
intermediate through HBM, so a cache read of ``N`` elements moves roughly 40N
bytes, and the full cache is rotated on every attention call.

This kernel reads each BF16 pair once, rotates it in float64 registers with the
same expression and the same float64 -> float32 -> BF16 rounding as the
reference, and writes it back once (4N bytes).  It may write in place.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_TOKENS = 16


@triton.jit(do_not_specialize=["num_tokens", "start_frame"])
def _rope_native_kernel(
    x_ptr,
    out_ptr,
    freqs_ptr,
    num_tokens,
    start_frame,
    x_stride_s,
    x_stride_h,
    out_stride_s,
    out_stride_h,
    freqs_stride_row,
    freqs_stride_col,
    tokens_per_frame: tl.constexpr,
    grid_width: tl.constexpr,
    heads: tl.constexpr,
    pairs: tl.constexpr,
    temporal_pairs: tl.constexpr,
    height_pairs: tl.constexpr,
    block_t: tl.constexpr,
    block_p: tl.constexpr,
):
    token = tl.program_id(0) * block_t + tl.arange(0, block_t)
    pair = tl.arange(0, block_p)
    mask = (token[:, None] < num_tokens) & (pair[None, :] < pairs)
    # The multiplier of pair j comes from the temporal, height or width table
    # according to j, indexed by the token's absolute frame, row or column --
    # the same concatenation the reference builds per model block.
    frame = start_frame + token // tokens_per_frame
    within_frame = token % tokens_per_frame
    row = within_frame // grid_width
    column = within_frame % grid_width
    position = tl.where(
        pair[None, :] < temporal_pairs,
        frame[:, None],
        tl.where(
            pair[None, :] < temporal_pairs + height_pairs,
            row[:, None],
            column[:, None],
        ),
    )
    freq_offset = position * freqs_stride_row + pair[None, :] * freqs_stride_col
    multiplier_real = tl.load(freqs_ptr + freq_offset, mask=mask, other=0.0)
    multiplier_imag = tl.load(freqs_ptr + freq_offset + 1, mask=mask, other=0.0)
    token_index = token.to(tl.int64)
    for head in range(heads):
        x_offset = (
            token_index[:, None] * x_stride_s + head * x_stride_h + 2 * pair[None, :]
        )
        x_real = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float64)
        x_imag = tl.load(x_ptr + x_offset + 1, mask=mask, other=0.0).to(tl.float64)
        # c10::complex<double>::operator*=, operand for operand.
        real = x_real * multiplier_real - x_imag * multiplier_imag
        imag = x_real * multiplier_imag + x_imag * multiplier_real
        out_offset = (
            token_index[:, None] * out_stride_s
            + head * out_stride_h
            + 2 * pair[None, :]
        )
        # A float64 -> BF16 cast in PyTorch narrows through float32 first.
        tl.store(
            out_ptr + out_offset,
            real.to(tl.float32).to(tl.bfloat16),
            mask=mask,
        )
        tl.store(
            out_ptr + out_offset + 1,
            imag.to(tl.float32).to(tl.bfloat16),
            mask=mask,
        )


def supports(x: torch.Tensor, out: torch.Tensor, freqs: torch.Tensor) -> bool:
    """Whether this kernel reproduces the reference for these tensors."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and out.device == x.device
        and freqs.dtype == torch.complex128
        and freqs.device == x.device
        and x.stride(-1) == 1
        and out.stride(-1) == 1
        and x.size(-1) % 2 == 0
        and freqs.size(1) == x.size(-1) // 2
    )


def rotate_frames(
    x: torch.Tensor,
    out: torch.Tensor,
    freqs: torch.Tensor,
    *,
    absolute_start_frame: int,
    tokens_per_frame: int,
    grid_width: int,
    temporal_pairs: int,
    height_pairs: int,
) -> None:
    """Rotate ``x`` ``[S, H, D]`` into ``out`` at consecutive absolute frames."""
    tokens, heads, head_dim = x.shape
    if tokens == 0:
        return
    freqs_real = torch.view_as_real(freqs)
    pairs = head_dim // 2
    grid = (triton.cdiv(tokens, BLOCK_TOKENS),)
    _rope_native_kernel[grid](
        x,
        out,
        freqs_real,
        tokens,
        absolute_start_frame,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        freqs_real.stride(0),
        freqs_real.stride(1),
        tokens_per_frame,
        grid_width,
        heads,
        pairs,
        temporal_pairs,
        height_pairs,
        BLOCK_TOKENS,
        triton.next_power_of_2(pairs),
        num_warps=4,
    )


__all__ = ["rotate_frames", "supports"]
