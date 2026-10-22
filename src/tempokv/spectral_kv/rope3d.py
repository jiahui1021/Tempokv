"""Three-axis RoPE for full tokens and fractional-position summary tokens.

Integer positions use the provided runtime frequency entries verbatim.  Between
entries, rotations follow the continuous phase through the adjacent entries;
the position-one phase resolves wrapping.  Outside the table, the endpoint is
continued using that position-one angular increment.  This continuation assumes
the constant angular frequencies used by Self-Forcing/Wan.  Frequencies aliased
by more than pi per position cannot be recovered from a sampled complex table.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from .model_spec import ModelSpec


def full_positions(
    frame_idx0: int,
    frames: int,
    hw: tuple[int, int],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return float64 coordinates in frame/height/width (row-major) order."""
    for value, name in ((frame_idx0, "frame_idx0"), (frames, "frames")):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if frame_idx0 < 0 or frames < 1:
        raise ValueError("frame_idx0 must be nonnegative and frames must be positive")
    if len(hw) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in hw):
        raise ValueError("hw must contain positive integer height and width")
    axes = (
        torch.arange(frame_idx0, frame_idx0 + frames, dtype=torch.float64, device=device),
        torch.arange(hw[0], dtype=torch.float64, device=device),
        torch.arange(hw[1], dtype=torch.float64, device=device),
    )
    return tuple(axis.reshape(-1) for axis in torch.meshgrid(*axes, indexing="ij"))


def _positions(value: torch.Tensor, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != length:
        raise ValueError("each position tensor must have shape [L]")
    if value.dtype == torch.bool or value.is_complex() or not (value.is_floating_point() or value.dtype in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    )):
        raise ValueError("positions must be real numeric tensors")
    if not torch.isfinite(value).all():
        raise ValueError("positions must be finite")
    return value.to(device=device, dtype=dtype)


def _runtime_axis(table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Interpolate rotations while preserving every in-table integer entry."""
    count = table.shape[0]
    integral = positions == torch.floor(positions)
    in_table = (positions >= 0) & (positions < count)
    exact = integral & in_table
    if bool(exact.all()):
        return table.index_select(0, positions.to(torch.long))
    if count < 2:
        raise ValueError("fractional or extrapolated positions require at least two frequency rows")
    # Work with a bounded integer anchor so even very large finite coordinates
    # do not overflow int64 when selecting a table entry.
    lower = positions.floor().clamp(0, count - 1).to(torch.long)
    upper = (lower + 1).clamp(max=count - 1)
    anchor = table.index_select(0, lower)
    next_value = table.index_select(0, upper)
    omega = torch.angle(table[1] * table[0].conj())
    step = torch.angle(next_value * anchor.conj())
    step = step + (2 * math.pi) * torch.round((omega - step) / (2 * math.pi))
    interior = (positions >= 0) & (positions < count - 1)
    step = torch.where(interior[:, None], step, omega[None, :])
    phase = (positions - lower.to(positions.dtype))[:, None] * step
    rotated = anchor * torch.polar(torch.ones_like(phase), phase)
    return torch.where(exact[:, None], anchor, rotated)


def _multipliers(
    positions: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spec: ModelSpec,
    freqs: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    if freqs is not None:
        if not isinstance(freqs, torch.Tensor) or freqs.ndim != 2 or not freqs.is_complex():
            raise ValueError("freqs must be a complex tensor with shape [positions, head_dim / 2]")
        if freqs.shape[0] < 1 or freqs.shape[1] != spec.head_dim // 2:
            raise ValueError("freqs shape must be [positive positions, head_dim / 2]")
        if freqs.dtype not in (torch.complex64, torch.complex128) or not torch.isfinite(freqs).all():
            raise ValueError("freqs must contain finite complex64 or complex128 rotations")
        tolerance = 2e-5 if freqs.dtype == torch.complex64 else 1e-10
        if not torch.allclose(freqs.abs(), torch.ones_like(freqs.real), atol=tolerance, rtol=tolerance):
            raise ValueError("freqs must contain unit-magnitude rotations")
        table = freqs.to(device=device, dtype=complex_dtype)
        axes = table.split(tuple(size // 2 for size in spec.rope_split), dim=1)
        return torch.cat([_runtime_axis(axis, pos) for axis, pos in zip(axes, positions)], dim=-1)
    axes = []
    for size, pos in zip(spec.rope_split, positions):
        omega = 1.0 / torch.pow(
            spec.rope_theta,
            torch.arange(0, size, 2, dtype=dtype, device=device).div(size),
        )
        phase = pos[:, None] * omega[None, :]
        axes.append(torch.polar(torch.ones_like(phase), phase))
    return torch.cat(axes, dim=-1)


def _rotate(
    x: torch.Tensor,
    pos_t: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    spec: ModelSpec,
    *,
    freqs: torch.Tensor | None,
    precision: str,
    inverse: bool,
) -> torch.Tensor:
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or not x.is_floating_point():
        raise ValueError("x must be a real floating tensor with shape [B, L, H, D]")
    if x.shape[2:] != (spec.heads, spec.head_dim):
        raise ValueError("x heads and head dimension must match ModelSpec")
    dtype = torch.float64 if precision == "float64" else torch.float32
    positions = tuple(_positions(p, x.shape[1], x.device, dtype) for p in (pos_t, pos_h, pos_w))
    multiplier = _multipliers(positions, spec, freqs, dtype, x.device)
    if inverse:
        multiplier = multiplier.conj()
    paired = x.to(dtype=dtype).contiguous().reshape(*x.shape[:-1], spec.head_dim // 2, 2)
    rotated = torch.view_as_complex(paired) * multiplier[None, :, None, :]
    return torch.view_as_real(rotated).flatten(-2).to(dtype=x.dtype)


def rope_apply(
    x: torch.Tensor,
    pos_t: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    spec: ModelSpec,
    *,
    freqs: torch.Tensor | None = None,
    precision: str = "float64",
) -> torch.Tensor:
    """Rotate ``[B,L,H,D]`` tokens using runtime or theta-derived frequencies."""
    return _rotate(x, pos_t, pos_h, pos_w, spec, freqs=freqs, precision=precision, inverse=False)


def rope_invert(
    x: torch.Tensor,
    pos_t: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    spec: ModelSpec,
    *,
    freqs: torch.Tensor | None = None,
    precision: str = "float64",
) -> torch.Tensor:
    """Undo :func:`rope_apply` at the same token coordinates."""
    return _rotate(x, pos_t, pos_h, pos_w, spec, freqs=freqs, precision=precision, inverse=True)
