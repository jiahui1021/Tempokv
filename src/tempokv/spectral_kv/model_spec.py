"""Runtime dimensions for the independent Spectral KV experiment suite.

There are deliberately no checkpoint or image-resolution presets here.  In Wan,
the three RoPE segments contain complex channels in the ratio implied by the
*runtime* head dimension, not by a nominal model name.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class ModelSpec:
    layers: int
    heads: int
    head_dim: int
    frame_hw: tuple[int, int]
    chunk_frames: int
    rope_split: tuple[int, int, int]
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in ("layers", "heads", "head_dim", "chunk_frames"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if len(self.frame_hw) != 2:
            raise ValueError("frame_hw must contain height and width")
        hw = tuple(_positive_integer(v, "frame_hw") for v in self.frame_hw)
        object.__setattr__(self, "frame_hw", hw)
        if len(self.rope_split) != 3:
            raise ValueError("rope_split must contain temporal, height and width dimensions")
        split = tuple(_positive_integer(v, "rope_split") for v in self.rope_split)
        if any(v % 2 for v in split) or sum(split) != self.head_dim:
            raise ValueError("rope_split must have even positive dimensions summing to head_dim")
        object.__setattr__(self, "rope_split", split)
        if not isinstance(self.rope_theta, (int, float)) or isinstance(self.rope_theta, bool):
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        object.__setattr__(self, "rope_theta", float(self.rope_theta))

    @classmethod
    def from_runtime(
        cls,
        model: Any,
        grid_sizes: Any = None,
        *,
        frame_hw: tuple[int, int] | None = None,
        chunk_frames: int | None = None,
    ) -> "ModelSpec":
        """Read Wan dimensions from a model and its actual token grid.

        ``grid_sizes`` is the model's ``[batch, 3]`` (frames, height, width)
        tensor or sequence.  A spec describes one grid, so mixed batch grids
        fail explicitly.  Without it, ``frame_hw`` must be supplied; the chunk
        length can then come from ``model.num_frame_per_block``.  An explicit
        chunk length may describe a configured chunk larger than this call's
        grid, for example the initial single-frame chunk.
        """
        blocks = getattr(model, "blocks", None)
        layers = len(blocks) if blocks is not None else getattr(model, "num_layers", None)
        attention = getattr(blocks[0], "self_attn", None) if blocks is not None and len(blocks) else None
        heads = getattr(model, "num_heads", None)
        if heads is None and attention is not None:
            heads = getattr(attention, "num_heads", None)
        heads = _positive_integer(heads, "runtime heads")
        head_dim = getattr(model, "head_dim", None)
        if head_dim is None and attention is not None:
            head_dim = getattr(attention, "head_dim", None)
        if head_dim is None:
            dim = getattr(model, "dim", None)
            if dim is None and attention is not None:
                dim = getattr(attention, "dim", None)
            dim = _positive_integer(dim, "runtime model dimension")
            if dim % heads:
                raise ValueError("runtime model dimension must be divisible by heads")
            head_dim = dim // heads
        head_dim = _positive_integer(head_dim, "runtime head_dim")
        if head_dim % 2:
            raise ValueError("runtime head_dim must be even")
        if grid_sizes is not None:
            rows = grid_sizes.tolist() if hasattr(grid_sizes, "tolist") else list(grid_sizes)
            if not rows or any(not hasattr(row, "__len__") or len(row) != 3 for row in rows):
                raise ValueError("grid_sizes must have shape [batch, 3]")
            grids = [tuple(_positive_integer(v, "runtime grid") for v in row) for row in rows]
            if any(grid != grids[0] for grid in grids[1:]):
                raise ValueError("one ModelSpec cannot describe mixed runtime grids")
            grid_frames, grid_h, grid_w = grids[0]
            if frame_hw is not None and tuple(frame_hw) != (grid_h, grid_w):
                raise ValueError("frame_hw disagrees with grid_sizes")
            frame_hw = (grid_h, grid_w)
            if chunk_frames is None:
                chunk_frames = grid_frames
        if frame_hw is None:
            raise ValueError("runtime grid_sizes or frame_hw is required")
        if chunk_frames is None:
            chunk_frames = getattr(model, "num_frame_per_block", None)
        complex_channels = head_dim // 2
        spatial = complex_channels // 3
        split = (2 * (complex_channels - 2 * spatial), 2 * spatial, 2 * spatial)
        theta = getattr(model, "rope_theta", 10000.0)
        return cls(layers, heads, head_dim, frame_hw, chunk_frames, split, theta)


def from_runtime(
    model: Any,
    grid_sizes: Any = None,
    *,
    frame_hw: tuple[int, int] | None = None,
    chunk_frames: int | None = None,
) -> ModelSpec:
    """Convenience alias for :meth:`ModelSpec.from_runtime`."""
    return ModelSpec.from_runtime(
        model, grid_sizes, frame_hw=frame_hw, chunk_frames=chunk_frames
    )
