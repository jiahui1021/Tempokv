"""Unnormalised Walsh-Hadamard transform; CUDA uses the upstream dependency."""
import torch


def reference_hadamard(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    width = x.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    y = x.float()
    step = 1
    while step < width:
        pair = y.reshape(*y.shape[:-1], -1, 2, step)
        a, b = pair.unbind(-2)
        y = torch.stack((a + b, a - b), dim=-2).reshape_as(y)
        step *= 2
    return (y * scale).to(x.dtype)


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if x.is_cuda:
        try:
            from fast_hadamard_transform import hadamard_transform as fast_transform
        except ImportError as error:
            raise RuntimeError("AquaKV HIGGS on CUDA requires fast-hadamard-transform") from error
        return fast_transform(x, scale=scale)
    return reference_hadamard(x, scale)
