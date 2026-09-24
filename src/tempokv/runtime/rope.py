"""Read-time causal RoPE for pre-RoPE KV caches.

Reconstructed K is rotated at each frame's absolute position, one contiguous
span at a time.  On CUDA the native (complex128) rotation runs as one Triton
kernel; ``TEMPOKV_ROPE_BACKEND=torch`` keeps the PyTorch path for A/B runs.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

import torch

from .cache import CacheSpan


ROPE_BACKEND_ENV = "TEMPOKV_ROPE_BACKEND"

# Geometries whose fused rotation was compared against the reference in this
# process; see _verify_fused_rotation.
_VERIFIED_FUSED: set[tuple[Any, ...]] = set()


def _fused_rotation(
    x: torch.Tensor, result: torch.Tensor, freqs: torch.Tensor, precision: str
) -> Any | None:
    """The fused kernel module when it reproduces this rotation, else None."""
    if os.environ.get(ROPE_BACKEND_ENV, "auto") == "torch":
        return None
    if precision != "native" or not x.is_cuda:
        return None
    from . import rope_triton

    return rope_triton if rope_triton.supports(x, result, freqs) else None


def _rotate_frames(
    x: torch.Tensor,
    result: torch.Tensor,
    *,
    batch_index: int,
    token_start: int,
    num_frames: int,
    absolute_start_frame: int,
    tokens_per_frame: int,
    height: int,
    width: int,
    block_frames: int,
    temporal: torch.Tensor,
    height_freqs: torch.Tensor,
    width_freqs: torch.Tensor,
    precision: str,
) -> None:
    heads = x.size(2)
    work_dtype = torch.float32 if precision == "complex64" else torch.float64
    # native follows upstream causal_rope_apply (complex128); complex64 matches the
    # legacy cached-K reader.
    for relative_frame in range(0, num_frames, block_frames):
        current_frames = min(block_frames, num_frames - relative_frame)
        block_start = token_start + relative_frame * tokens_per_frame
        block_end = block_start + current_frames * tokens_per_frame
        complex_x = torch.view_as_complex(
            x[batch_index, block_start:block_end]
            .to(work_dtype)
            .reshape(block_end - block_start, heads, -1, 2)
        )
        absolute_frame = absolute_start_frame + relative_frame
        multipliers = torch.cat(
            [
                temporal[absolute_frame:absolute_frame + current_frames]
                .view(current_frames, 1, 1, -1)
                .expand(current_frames, height, width, -1),
                height_freqs[:height]
                .view(1, height, 1, -1)
                .expand(current_frames, height, width, -1),
                width_freqs[:width]
                .view(1, 1, width, -1)
                .expand(current_frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(block_end - block_start, 1, -1).to(dtype=complex_x.dtype)
        rotated = torch.view_as_real(complex_x * multipliers).flatten(2)
        result[batch_index, block_start:block_end].copy_(rotated.to(dtype=x.dtype))


def _verify_fused_rotation(
    fused: Any,
    freqs: torch.Tensor,
    *,
    device: torch.device,
    heads: int,
    head_dim: int,
    height: int,
    width: int,
    block_frames: int,
    temporal: torch.Tensor,
    height_freqs: torch.Tensor,
    width_freqs: torch.Tensor,
) -> None:
    """Check the fused kernel bit-for-bit against the reference once per geometry."""
    key = (str(device), heads, head_dim, height, width)
    if key in _VERIFIED_FUSED:
        return
    tokens_per_frame = height * width
    frames = min(3, temporal.shape[0])
    generator = torch.Generator(device=device).manual_seed(0)
    probe = torch.randn(
        1,
        frames * tokens_per_frame,
        heads,
        head_dim,
        generator=generator,
        device=device,
    ).to(torch.bfloat16)
    channels = head_dim // 2
    for start_frame in (0, temporal.shape[0] - frames):
        expected = torch.empty_like(probe)
        _rotate_frames(
            probe,
            expected,
            batch_index=0,
            token_start=0,
            num_frames=frames,
            absolute_start_frame=start_frame,
            tokens_per_frame=tokens_per_frame,
            height=height,
            width=width,
            block_frames=block_frames,
            temporal=temporal,
            height_freqs=height_freqs,
            width_freqs=width_freqs,
            precision="native",
        )
        actual = torch.empty_like(probe)
        fused.rotate_frames(
            probe[0],
            actual[0],
            freqs,
            absolute_start_frame=start_frame,
            tokens_per_frame=tokens_per_frame,
            grid_width=width,
            temporal_pairs=channels - 2 * (channels // 3),
            height_pairs=channels // 3,
        )
        if not torch.equal(actual, expected):
            mismatched = int((actual != expected).sum())
            raise RuntimeError(
                f"the Triton RoPE kernel differs from the float64 reference in "
                f"{mismatched} values at frame {start_frame}; set "
                f"{ROPE_BACKEND_ENV}=torch to use the reference"
            )
    _VERIFIED_FUSED.add(key)


def apply_rope_by_span(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    spans: Sequence[CacheSpan] | Iterable[CacheSpan],
    *,
    precision: str = "native",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate a pre-RoPE ``[B, S, H, D]`` cache read span by span.

    ``out`` may alias ``x`` for an in-place rotation.
    """
    spans = list(spans)
    channels = x.size(3) // 2
    temporal, height_freqs, width_freqs = freqs.split(
        [channels - 2 * (channels // 3), channels // 3, channels // 3], dim=1
    )
    result = torch.empty_like(x) if out is None else out
    fused = _fused_rotation(x, result, freqs, precision)
    for batch_index, (model_block_frames, height, width) in enumerate(
        grid_sizes.tolist()
    ):
        tokens_per_frame = height * width
        block_frames = max(1, int(model_block_frames))
        if fused is not None:
            _verify_fused_rotation(
                fused,
                freqs,
                device=x.device,
                heads=x.size(2),
                head_dim=x.size(3),
                height=height,
                width=width,
                block_frames=block_frames,
                temporal=temporal,
                height_freqs=height_freqs,
                width_freqs=width_freqs,
            )
        token_start = 0
        for span in spans:
            if fused is not None:
                # The kernel has no bounds check, so a frame past the table
                # would read unrelated memory instead of failing.
                if span.global_start_frame + span.num_frames > temporal.shape[0]:
                    raise ValueError(
                        f"RoPE table supports {temporal.shape[0]} frames, "
                        f"requested [{span.global_start_frame}, "
                        f"{span.global_start_frame + span.num_frames})"
                    )
                token_end = token_start + span.num_frames * tokens_per_frame
                fused.rotate_frames(
                    x[batch_index, token_start:token_end],
                    result[batch_index, token_start:token_end],
                    freqs,
                    absolute_start_frame=span.global_start_frame,
                    tokens_per_frame=tokens_per_frame,
                    grid_width=width,
                    temporal_pairs=channels - 2 * (channels // 3),
                    height_pairs=channels // 3,
                )
                token_start = token_end
                continue
            _rotate_frames(
                x,
                result,
                batch_index=batch_index,
                token_start=token_start,
                num_frames=span.num_frames,
                absolute_start_frame=span.global_start_frame,
                tokens_per_frame=tokens_per_frame,
                height=height,
                width=width,
                block_frames=block_frames,
                temporal=temporal,
                height_freqs=height_freqs,
                width_freqs=width_freqs,
                precision=precision,
            )
            token_start += span.num_frames * tokens_per_frame
    return result


def causal_rope_apply_long_input(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
    *,
    precision: str = "native",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate one contiguous pre-RoPE cache read starting at ``start_frame``."""
    tokens_per_frame = int(grid_sizes[0][1]) * int(grid_sizes[0][2])
    frames = x.shape[1] // tokens_per_frame
    return apply_rope_by_span(
        x,
        grid_sizes,
        freqs,
        [CacheSpan(0, int(start_frame), frames)],
        precision=precision,
        out=out,
    )


__all__ = ["ROPE_BACKEND_ENV", "apply_rope_by_span", "causal_rope_apply_long_input"]
