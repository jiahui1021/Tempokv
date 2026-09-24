"""Offline QVG-style residual vector quantization for TempoKV experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .quantization import roundtrip


@dataclass(frozen=True)
class ResidualVQCodebook:
    """One residual-VQ stage with a separate codebook per layer and head."""

    centroids: torch.Tensor

    @property
    def num_centroids(self) -> int:
        return int(self.centroids.shape[2])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[3])

    def to(self, device: torch.device | str) -> "ResidualVQCodebook":
        return ResidualVQCodebook(self.centroids.to(device))

    def parameter_bytes(self) -> int:
        return self.centroids.numel() * self.centroids.element_size()

    def physical_index_bits_per_value(self) -> float:
        """Current QVG-compatible UINT8 index storage, amortized over D."""
        return 8.0 / self.dim

    def logical_index_bits_per_value(self) -> float:
        return math.ceil(math.log2(self.num_centroids)) / self.dim


def _squared_distances(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    point_norm = points.square().sum(dim=1, keepdim=True)
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    return (point_norm + center_norm - 2.0 * points @ centers.T).clamp_min_(0.0)


@torch.no_grad()
def fit_residual_vq_codebook(
    samples: torch.Tensor,
    *,
    num_centroids: int,
    iterations: int = 6,
    max_points: int = 4096,
    seed: int = 0,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> ResidualVQCodebook:
    """Fit deterministic per-layer/head Lloyd codebooks on ``[L,...,H,D]`` samples."""
    layers = int(samples.shape[0])
    heads = int(samples.shape[-2])
    dim = int(samples.shape[-1])

    output = torch.empty(
        layers,
        heads,
        num_centroids,
        dim,
        dtype=storage_dtype,
        device=samples.device,
    )
    for layer in range(layers):
        for head in range(heads):
            points = samples[layer, ..., head, :].reshape(-1, dim).float()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1009 * layer + 9176 * head)
            order = torch.randperm(int(points.shape[0]), generator=generator)
            if points.device.type != "cpu":
                order = order.to(points.device)
            points = points.index_select(0, order[: min(max_points, len(order))])
            centers = points[:num_centroids].clone()
            for _ in range(iterations):
                assignment = _squared_distances(points, centers).argmin(dim=1)
                counts = torch.bincount(assignment, minlength=num_centroids)
                sums = torch.zeros_like(centers)
                sums.index_add_(0, assignment, points)
                updated = centers.clone()
                occupied = counts > 0
                updated[occupied] = sums[occupied] / counts[occupied, None]
                if torch.equal(updated, centers):
                    break
                centers = updated
            output[layer, head] = centers.to(storage_dtype)
    return ResidualVQCodebook(output)


@torch.no_grad()
def nearest_centroid(
    values: torch.Tensor,
    codebook: ResidualVQCodebook,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected centroids and UINT8-compatible indices for ``[L,...,H,D]``."""
    layers = int(values.shape[0])
    heads = int(values.shape[-2])
    dim = int(values.shape[-1])
    # Larger codebooks would silently wrap in the UINT8 indices.
    if codebook.num_centroids > 256:
        raise ValueError("QVG-compatible UINT8 indices support at most 256 centroids")

    flat = values.float().reshape(layers, -1, heads, dim)
    selected = torch.empty_like(flat)
    indices = torch.empty(
        layers,
        flat.shape[1],
        heads,
        dtype=torch.uint8,
        device=values.device,
    )
    centers = codebook.centroids.to(device=values.device, dtype=torch.float32)
    for layer in range(layers):
        for head in range(heads):
            assignment = _squared_distances(
                flat[layer, :, head], centers[layer, head]
            ).argmin(dim=1)
            indices[layer, :, head] = assignment.to(torch.uint8)
            selected[layer, :, head] = centers[layer, head].index_select(0, assignment)
    return selected.reshape(values.shape), indices.reshape(*values.shape[:-1])


@torch.no_grad()
def residual_vq_roundtrip(
    residual: torch.Tensor,
    codebook: ResidualVQCodebook,
    *,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype = torch.bfloat16,
    symmetric: bool = False,
) -> torch.Tensor:
    """Reconstruct one TempoKV residual as centroid plus low-bit scalar tail."""
    selected, _ = nearest_centroid(residual, codebook)
    tail = residual.float() - selected
    tail_rebuilt = roundtrip(
        tail,
        bits=bits,
        group_size=group_size,
        scale_dtype=scale_dtype,
        symmetric=symmetric,
    )
    return selected + tail_rebuilt


__all__ = [
    "ResidualVQCodebook",
    "fit_residual_vq_codebook",
    "nearest_centroid",
    "residual_vq_roundtrip",
]
