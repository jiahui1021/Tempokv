"""Read-time causal RoPE for pre-RoPE KV caches.

The cache stores normalized K *before* the rotary embedding so a TempoKV
temporal predictor only ever models content change, never the positional
rotation.  Attention therefore has to rotate the reconstructed K on the way
out, using each frame's **absolute** position on the model timeline.

When the resident cache is contiguous this is a single call with one start
frame.  Once eviction leaves a sink/recent gap the resident frames no longer
form one arithmetic sequence, so each run is rotated with its own absolute
start and only then concatenated: rotating `[0, 1, 100, 101]` as if it were
`[0, 1, 2, 3]` would silently corrupt every post-gap key.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .cache import CacheSpan


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
) -> None:
    heads = x.size(2)
    if absolute_start_frame + num_frames > temporal.shape[0]:
        raise ValueError(
            f"RoPE table supports {temporal.shape[0]} frames, requested "
            f"[{absolute_start_frame}, {absolute_start_frame + num_frames})"
        )
    # Rotate in complex128, matching upstream `causal_rope_apply`: the RoPE
    # table is built in float64 and downcasting it to complex64 here would make
    # a handful of bf16 keys per frame differ from the reference
    # implementation, which an autoregressive rollout then feeds back into
    # itself. Temporaries stay bounded by processing one model block at a time
    # rather than the whole history, so for Wan2.1-1.3B this costs about
    # 114 MiB transiently per layer call instead of 57 MiB.
    for relative_frame in range(0, num_frames, block_frames):
        current_frames = min(block_frames, num_frames - relative_frame)
        block_start = token_start + relative_frame * tokens_per_frame
        block_end = block_start + current_frames * tokens_per_frame
        complex_x = torch.view_as_complex(
            x[batch_index, block_start:block_end]
            .to(torch.float64)
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


def apply_rope_by_span(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    spans: Sequence[CacheSpan] | Iterable[CacheSpan],
) -> torch.Tensor:
    """Rotate a pre-RoPE cache read, one absolute-position span at a time.

    ``x`` is ``[B, S, H, D]`` holding the spans back to back in the order
    given, which is the order :meth:`ChunkedKVCache.spans` returns.
    """
    spans = list(spans)
    if not spans:
        raise ValueError("RoPE requires at least one cache span")
    channels = x.size(3) // 2
    temporal, height_freqs, width_freqs = freqs.split(
        [channels - 2 * (channels // 3), channels // 3, channels // 3], dim=1
    )
    result = torch.empty_like(x)
    for batch_index, (model_block_frames, height, width) in enumerate(
        grid_sizes.tolist()
    ):
        tokens_per_frame = height * width
        if x.shape[1] % tokens_per_frame:
            raise ValueError("cached sequence is not frame aligned")
        total_frames = sum(span.num_frames for span in spans)
        if total_frames * tokens_per_frame != x.shape[1]:
            raise ValueError(
                f"spans cover {total_frames} frames but the cache read holds "
                f"{x.shape[1] // tokens_per_frame}"
            )
        block_frames = max(1, int(model_block_frames))
        token_start = 0
        for span in spans:
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
            )
            token_start += span.num_frames * tokens_per_frame
    return result


def causal_rope_apply_long_input(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor:
    """Rotate one contiguous pre-RoPE cache read starting at ``start_frame``."""
    tokens_per_frame = int(grid_sizes[0][1]) * int(grid_sizes[0][2])
    if tokens_per_frame <= 0:
        raise ValueError("grid sizes must be positive")
    if x.shape[1] % tokens_per_frame:
        raise ValueError("cached sequence is not frame aligned")
    frames = x.shape[1] // tokens_per_frame
    return apply_rope_by_span(
        x,
        grid_sizes,
        freqs,
        [CacheSpan(0, int(start_frame), frames)],
    )


__all__ = ["apply_rope_by_span", "causal_rope_apply_long_input"]
