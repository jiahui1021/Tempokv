"""E1 statistics; dense features are consumed and released one clean frame at a time."""
from __future__ import annotations

import torch

from .transforms import radial_mask, spatial_fft, temporal_fft


def spatial_energy(tensor: torch.Tensor, cutoffs, eps: float = 1e-12):
    power = spatial_fft(tensor).abs().square().sum(dim=-1)
    total = power.sum(dim=(2, 3))  # B,F,heads
    zero = total <= eps
    ratios = {}
    for cutoff in cutoffs:
        mask = radial_mask(tensor.shape[2], tensor.shape[3], cutoff, device=tensor.device)
        low = (power * mask[None, None, :, :, None]).sum(dim=(2, 3))
        ratios[cutoff] = (low / (total + eps)).masked_fill(zero, float("nan"))
    return ratios, total, zero


def temporal_energy(tensor: torch.Tensor, cutoffs, eps: float = 1e-12):
    power = temporal_fft(tensor).abs().square().sum(dim=(1, 3))  # T,heads
    total = power.sum(dim=0)
    frequency = torch.fft.fftfreq(tensor.shape[0], device=tensor.device)
    ratios = torch.stack([
        (power[frequency.abs() <= cutoff].sum(dim=0) / (total + eps))
        .masked_fill(total <= eps, float("nan")) for cutoff in cutoffs
    ])
    return frequency, power, ratios, total <= eps


def cosine_similarity(a: torch.Tensor | None, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-head cosine on [H,W,heads,D]; zero vectors are unmeasured."""
    if a is None or a.shape != b.shape:
        return torch.full((b.shape[-2],), float("nan"))
    a, b = a.flatten(0, 1).float(), b.flatten(0, 1).float()
    denominator = a.square().sum(dim=(0, 2)).sqrt() * b.square().sum(dim=(0, 2)).sqrt()
    return ((a * b).sum(dim=(0, 2)) / denominator.clamp_min(eps)).masked_fill(
        denominator <= eps, float("nan")
    )
