"""Separable spatial projections for suite B's Spectral KV experiments.

The DCT basis has frequency on its first axis.  Projection matrices are CPU
float32 tensors; applying them uses float32 arithmetic on the input device.
Public constructors return copies so callers cannot corrupt the cached bases.
"""
from __future__ import annotations

from functools import lru_cache
import math
from numbers import Integral

import torch


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _dimensions(n: int, c: int) -> tuple[int, int]:
    n, c = _positive_integer(n, "n"), _positive_integer(c, "c")
    if c > n:
        raise ValueError("compression cannot expand an axis: c must not exceed n")
    return n, c


@lru_cache(maxsize=128)
def _dct_matrix(n: int) -> torch.Tensor:
    # Evaluate cosines in float64 before storing the reusable float32 basis.
    position = torch.arange(n, dtype=torch.float64, device="cpu") + 0.5
    frequency = torch.arange(n, dtype=torch.float64, device="cpu")[:, None]
    basis = torch.cos(math.pi * frequency * position / n) * math.sqrt(2.0 / n)
    basis[0] *= 1.0 / math.sqrt(2.0)
    return basis.float()


def dct_matrix(n: int) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix ``[n, n]`` (rows = frequency)."""
    return _dct_matrix(_positive_integer(n, "n")).clone()


def _validate_method(kind: str, taper: str | None) -> None:
    if kind not in ("dct", "avg"):
        raise ValueError("kind must be 'dct' or 'avg'")
    if taper not in (None, "hann"):
        raise ValueError("taper must be None or 'hann'")
    if kind == "avg" and taper is not None:
        raise ValueError("taper is defined only for the DCT projection")


@lru_cache(maxsize=256)
def _projection(kind: str, n: int, c: int, taper: str | None) -> torch.Tensor:
    if c == n:
        # Taper is an ablation of truncation, not a filter at full resolution.
        return torch.eye(n, dtype=torch.float32, device="cpu")
    if kind == "dct":
        retained = _dct_matrix(n)[:c]
        if taper == "hann":
            # The descending half-Hann reaches zero at the first omitted bin.
            # All retained bins remain nonzero, and the DC coefficient stays 1.
            frequency = torch.arange(c, dtype=torch.float32, device="cpu")
            weights = 0.5 * (1.0 + torch.cos(math.pi * frequency / c))
            retained = retained * weights[:, None]
        return math.sqrt(c / n) * (_dct_matrix(c).T @ retained)

    # Source cells occupy [j, j+1); target cells occupy [i*n/c, (i+1)*n/c).
    # Fractional overlap handles non-divisible grids without dropping or
    # duplicating edge cells, and each output remains a normalized average.
    source_start = torch.arange(n, dtype=torch.float64, device="cpu")[None]
    target_start = torch.arange(c, dtype=torch.float64, device="cpu")[:, None] * (n / c)
    overlap = (
        torch.minimum(source_start + 1.0, target_start + n / c)
        - torch.maximum(source_start, target_start)
    ).clamp_min(0.0)
    return (overlap / (n / c)).float()


def make_projection(kind: str, n: int, c: int, taper: str | None = None) -> torch.Tensor:
    """Return a constant-preserving ``[c, n]`` low-pass/downsampling matrix.

    ``dct`` uses ``sqrt(c/n) * D_c.T @ D_n[:c]``.  ``hann`` multiplies its
    retained coefficient ``k`` by ``(1 + cos(pi*k/c))/2``: DC is unchanged,
    and the first omitted frequency would have zero weight.  ``avg`` uses
    normalized fractional cell overlaps and does not accept a taper.

    Expansion is rejected.  At ``c == n`` both kinds return exact identity;
    the DCT taper is bypassed so full resolution is an exact baseline.
    """
    _validate_method(kind, taper)
    n, c = _dimensions(n, c)
    return _projection(kind, n, c, taper).clone()


def cell_centers(n: int, c: int) -> torch.Tensor:
    """Return target-cell centers in the original zero-based coordinate grid."""
    n, c = _dimensions(n, c)
    return (torch.arange(c, dtype=torch.float32, device="cpu") + 0.5) * (n / c) - 0.5


def _validate_grid(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 5:
        raise ValueError("grid must have shape [F,H,W,heads,D]")
    if not x.is_floating_point():
        raise ValueError("grid must have a real floating-point dtype")
    if min(x.shape) < 1:
        raise ValueError("all grid dimensions must be positive")


def _validate_projection(p: torch.Tensor, n: int, name: str) -> None:
    if not isinstance(p, torch.Tensor) or p.ndim != 2 or p.shape[1] != n:
        raise ValueError(f"{name} must have shape [compressed,{n}]")
    if not p.is_floating_point():
        raise ValueError(f"{name} must have a real floating-point dtype")
    if p.shape[0] < 1 or p.shape[0] > n:
        raise ValueError(f"{name} must preserve or reduce its input axis")


def _apply_grid(x: torch.Tensor, ph: torch.Tensor, pw: torch.Tensor) -> torch.Tensor:
    # Disabling an enclosing autocast region is necessary to guarantee FP32
    # contractions; .float() alone would still be autocast back to BF16/FP16.
    with torch.autocast(device_type=x.device.type, enabled=False):
        ph = ph.to(device=x.device, dtype=torch.float32)
        pw = pw.to(device=x.device, dtype=torch.float32)
        y = torch.einsum("ah,fhwnd->fawnd", ph, x.float())
        y = torch.einsum("bw,fawnd->fabnd", pw, y)
    return y.to(dtype=x.dtype)


def compress_grid(x: torch.Tensor, Ph: torch.Tensor, Pw: torch.Tensor) -> torch.Tensor:
    """Project ``[F,H,W,heads,D]`` to ``[F,ch,cw,heads,D]`` in float32.

    The output retains the input dtype and device.  Matrices may reside on a
    different device and are copied to the input device for the contractions.
    """
    _validate_grid(x)
    _validate_projection(Ph, x.shape[1], "Ph")
    _validate_projection(Pw, x.shape[2], "Pw")
    return _apply_grid(x, Ph, Pw)


@lru_cache(maxsize=256)
def _reconstruction(kind: str, n: int, c: int, taper: str | None) -> torch.Tensor:
    # A minimum-norm right inverse reconstructs the retained DCT subspace.
    # For tapered DCT this also undoes the explicitly supplied taper.  It is
    # intended for offline checks, not for materializing the attention cache.
    p = _projection(kind, n, c, taper)
    return torch.linalg.pinv(p.double()).float()


def reconstruct_grid(
    y: torch.Tensor,
    original_hw: tuple[int, int],
    kind: str = "dct",
    taper: str | None = None,
) -> torch.Tensor:
    """Lift a compressed grid using the matching projection's right inverse.

    Untapered DCT reconstructs exactly any signal supported on the retained
    rectangular frequency band, up to floating-point error.  ``avg`` returns
    the minimum-norm grid consistent with its cell averages.  Both preserve
    constants.  Tapered reconstruction reverses the taper and can amplify
    noise near its cutoff, so this helper is for offline correctness checks.
    """
    _validate_grid(y)
    _validate_method(kind, taper)
    if not isinstance(original_hw, (tuple, list)) or len(original_hw) != 2:
        raise ValueError("original_hw must contain the original height and width")
    h, ch = _dimensions(original_hw[0], y.shape[1])
    w, cw = _dimensions(original_hw[1], y.shape[2])
    return _apply_grid(y, _reconstruction(kind, h, ch, taper), _reconstruction(kind, w, cw, taper))
