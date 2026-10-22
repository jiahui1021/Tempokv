"""Frequency coordinates are cycles/token, with orthonormal transforms."""
from __future__ import annotations

import torch


def grid_shape(tensor: torch.Tensor, grid_sizes) -> tuple[int, int, int]:
    grids = torch.as_tensor(grid_sizes).detach().cpu()
    if tensor.ndim != 4 or grids.shape != (tensor.shape[0], 3):
        raise ValueError("expected [B,S,heads,D] and grid_sizes=[B,3]")
    if grids.is_floating_point() and not torch.equal(grids, grids.round()):
        raise ValueError("grid sizes must be integers")
    if not torch.all(grids == grids[0]) or torch.any(grids <= 0):
        raise ValueError("spectral probe needs equal positive grids without padding")
    f, h, w = (int(x) for x in grids[0])
    if f * h * w != tensor.shape[1]:
        raise ValueError("sequence length does not match runtime F*H*W")
    return f, h, w


def spatial_layout(tensor: torch.Tensor, grid_sizes) -> torch.Tensor:
    f, h, w = grid_shape(tensor, grid_sizes)
    return tensor.reshape(tensor.shape[0], f, h, w, *tensor.shape[2:])


def spatial_fft(tensor: torch.Tensor) -> torch.Tensor:
    """Transform [B,F,H,W,heads,D] along H,W, never along sampled tokens."""
    if tensor.ndim != 6:
        raise ValueError("spatial FFT expects [B,F,H,W,heads,D]")
    return torch.fft.fft2(tensor.float(), dim=(2, 3), norm="ortho")


def temporal_fft(tensor: torch.Tensor) -> torch.Tensor:
    """Transform [T,positions,heads,D] at fixed spatial positions."""
    if tensor.ndim != 4 or tensor.shape[0] < 2:
        raise ValueError("temporal FFT requires at least two aligned frames")
    return torch.fft.fft(tensor.float(), dim=0, norm="ortho")


def radial_mask(h: int, w: int, cutoff: float, *, device=None) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.fftfreq(w, device=device)
    return fy[:, None].square() + fx[None, :].square() <= cutoff**2


def spatial_blocks(h: int, w: int, block_size: tuple[int, int]):
    """Yield id, row slice, column slice, and row-major token indices."""
    bh, bw = block_size
    if min(h, w, bh, bw) < 1:
        raise ValueError("grid and block dimensions must be positive")
    block_id = 0
    for row in range(0, h, bh):
        for col in range(0, w, bw):
            rows, cols = slice(row, min(row + bh, h)), slice(col, min(col + bw, w))
            tokens = tuple(y * w + x for y in range(rows.start, rows.stop)
                           for x in range(cols.start, cols.stop))
            yield block_id, rows, cols, tokens
            block_id += 1
