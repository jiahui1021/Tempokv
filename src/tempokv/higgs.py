"""Packed HIGGS for KV residuals and anchors.

Widths 2–4 use the pinned AquaKV grids. The remaining widths use Gaussian scalar grids.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import torch
from torch.nn.functional import pad

from higgs_kernels import for_device

from .quantization import _pack, _unpack, cuda_matmul_precision

_ENCODE_CHUNK_SIZE = 64
_DECODE_CHUNK_SIZE = 1024
_VECTOR_BITS = frozenset({2, 3, 4})
_GRID_DIR = Path(__file__).with_name("higgs_grids")
_decode_reads = 0


def record_cache_read() -> None:
    global _decode_reads
    _decode_reads += 1


def cache_reads() -> int:
    return _decode_reads


@lru_cache(maxsize=None)
def _grid(bits: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if bits in _VECTOR_BITS:
        path = _GRID_DIR / f"EDEN2-{1 << (2 * bits)}.pt"
        grid = torch.load(path, map_location="cpu", weights_only=True).to(device)
        return grid, torch.linalg.norm(grid, dim=1).square()
    with (_GRID_DIR / "gaussian_scalar.json").open(encoding="utf-8") as stream:
        grid = torch.tensor(json.load(stream)[str(bits)], device=device, dtype=torch.float32)
    return grid, (grid[:-1] + grid[1:]) * 0.5


def _hadamard(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if value.is_cuda:
        try:
            from fast_hadamard_transform import hadamard_transform
        except ImportError:
            pass
        else:
            return hadamard_transform(value.contiguous(), scale=scale)
    result = value.float()
    step = 1
    while step < value.shape[-1]:
        a, b = result.reshape(*result.shape[:-1], -1, 2, step).unbind(-2)
        result = torch.stack((a + b, a - b), dim=-2).reshape_as(result)
        step *= 2
    return (result * scale).to(value.dtype)


def packed_widths(heads: int, dim: int, bits: int, group_size: int) -> tuple[int, int]:
    groups = (heads * dim + group_size - 1) // group_size
    # Both scalar and two-dimensional grids use this many bits per group.
    return groups * ((group_size * bits + 7) // 8), groups


@torch.no_grad()
def encode(
    value: torch.Tensor, bits: int, group_size: int, *, use_triton: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a [..., H, D] tensor. Return packed indices and FP32 scales."""
    heads, dim = value.shape[-2:]
    width, groups = packed_widths(heads, dim, bits, group_size)
    grid, search = _grid(bits, value.device)
    vector = bits in _VECTOR_BITS
    kernels = for_device(value.device, group_size) if use_triton else None
    if kernels is not None:
        rows = value if value.ndim == 4 else value.reshape(1, -1, heads, dim)
        codes, scales = kernels.quantize(
            rows, grid, search, group_size=group_size,
            code_bits=2 * bits if vector else bits, code_dim=2 if vector else 1,
        )
        return codes.reshape(*value.shape[:-2], width), scales.reshape(*value.shape[:-2], groups)
    rows = value.reshape(-1, heads * dim)
    codes = torch.empty((len(rows), width), device=value.device, dtype=torch.uint8)
    scales = torch.empty((len(rows), groups), device=value.device, dtype=torch.float32)
    with cuda_matmul_precision(value.device, "ieee"):
        for start in range(0, len(rows), _ENCODE_CHUNK_SIZE):
            stop = min(start + _ENCODE_CHUNK_SIZE, len(rows))
            batch = pad(rows[start:stop].float(), (0, groups * group_size - heads * dim))
            batch = batch.reshape(-1, groups, group_size)
            scale = torch.linalg.norm(batch, dim=-1)
            denominator = torch.where(scale == 0, 1.0, scale)
            rotated = _hadamard(batch) / denominator.unsqueeze(-1)
            if vector:
                pairs = rotated.reshape(-1, groups, group_size // 2, 2)
                indices = torch.argmax(2 * pairs @ grid.T - search, dim=-1)
            else:
                indices = torch.bucketize(rotated, search)
            codes[start:stop] = _pack(indices, 2 * bits if vector else bits).flatten(1)
            scales[start:stop] = scale
    return codes.reshape(*value.shape[:-2], width), scales.reshape(*value.shape[:-2], groups)


@torch.no_grad()
def decode(
    codes: torch.Tensor, scales: torch.Tensor, bits: int, group_size: int, heads: int, dim: int,
    *, use_triton: bool = True,
) -> torch.Tensor:
    """Recover [..., H, D] values from bit-packed codebook indices."""
    width, groups = packed_widths(heads, dim, bits, group_size)
    kernels = for_device(codes.device, group_size) if use_triton else None
    if kernels is not None:
        batch, tokens = codes.shape[:2] if codes.ndim == 3 else (1, codes.numel() // width)
        result = torch.empty((batch, tokens, heads, dim), device=codes.device, dtype=torch.float32)
        decode_into(
            codes.reshape(batch, tokens, width), scales.reshape(batch, tokens, groups),
            bits, group_size, result, kernels=kernels,
        )
        return result.reshape(*codes.shape[:-1], heads, dim)
    rows = codes.reshape(-1, groups, width // groups)
    scales = scales.reshape(-1, groups)
    grid, _ = _grid(bits, codes.device)
    vector = bits in _VECTOR_BITS
    code_bits = 2 * bits if vector else bits
    code_count = group_size // 2 if vector else group_size
    result = torch.empty((len(rows), heads * dim), device=codes.device, dtype=torch.float32)
    for start in range(0, len(rows), _DECODE_CHUNK_SIZE):
        stop = min(start + _DECODE_CHUNK_SIZE, len(rows))
        indices = _unpack(rows[start:stop], code_bits, code_count)
        rotated = grid[indices.long()].reshape(-1, groups, group_size)
        rotated = rotated * scales[start:stop].unsqueeze(-1)
        # Match AquaKV for its shipped grids. At finer scalar widths retain
        # FP32 so the transform does not discard the extra precision.
        restored = _hadamard(rotated.half() if vector else rotated, 1.0 / group_size).flatten(1)
        result[start:stop] = restored[:, :heads * dim]
    return result.reshape(*codes.shape[:-1], heads, dim)


def decode_into(codes, scales, bits, group_size, output, *, kernels, predictor=None):
    """Decode BSHD, optionally adding a FP32 prediction before the output cast."""
    grid, _ = _grid(bits, codes.device)
    vector = bits in _VECTOR_BITS
    kernels.dequantize_into(
        codes, scales, grid, output, group_size=group_size,
        code_dim=2 if vector else 1, code_bits=2 * bits if vector else bits,
        half_transform=vector, predictor=predictor,
    )


def roundtrip(
    value: torch.Tensor, bits: int, group_size: int, *, use_triton: bool = True,
) -> torch.Tensor:
    codes, scales = encode(value, bits, group_size, use_triton=use_triton)
    return decode(codes, scales, bits, group_size, *value.shape[-2:], use_triton=use_triton)
