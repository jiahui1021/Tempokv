"""Unnormalized Sylvester-Hadamard transform.

``fast_hadamard_transform.hadamard_transform`` is a CUDA kernel for this
transform: ``H @ H = n I`` with the Sylvester sign pattern, and ``scale``
multiplies the result. The Self-Forcing port runs the same transform in
PyTorch so calibration and tests do not need that CUDA extension.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply the unnormalized Walsh-Hadamard transform on the last dimension."""
    n = int(x.shape[-1])
    if n < 1 or n & (n - 1):
        raise ValueError("hadamard length must be a power of two")
    y = x.clone()
    h = 1
    while h < n:
        view = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :]
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    if float(scale) != 1.0:
        y = y * y.new_tensor(float(scale))
    return y
