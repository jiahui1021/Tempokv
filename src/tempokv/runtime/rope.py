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

The ``native`` (complex128) rotation of a CUDA BF16 read runs as one Triton
kernel (:mod:`tempokv.runtime.rope_triton`) instead of the per-block PyTorch
sequence below, which streams a float64 copy of the whole read through memory
several times.  ``TEMPOKV_ROPE_BACKEND=torch`` keeps the PyTorch sequence;
``triton`` requires the kernel; the default ``auto`` uses it whenever it
applies.
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


def rope_backend() -> str:
    backend = os.environ.get(ROPE_BACKEND_ENV, "auto")
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"{ROPE_BACKEND_ENV} must be auto, torch or triton")
    return backend


def _fused_rotation(
    x: torch.Tensor, result: torch.Tensor, freqs: torch.Tensor, precision: str
) -> Any | None:
    """The fused kernel module when it reproduces this rotation, else None."""
    backend = rope_backend()
    if backend == "torch":
        return None
    required = backend == "triton"
    if precision != "native" or not x.is_cuda:
        if required:
            raise ValueError(
                "the Triton RoPE kernel covers the native precision on CUDA only"
            )
        return None
    try:
        from . import rope_triton
    except ImportError:
        if required:
            raise
        return None
    if not rope_triton.supports(x, result, freqs):
        if required:
            raise ValueError(
                "the Triton RoPE kernel needs BF16 input and output with a "
                "contiguous head dimension and a complex128 RoPE table"
            )
        return None
    return rope_triton


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
    if absolute_start_frame + num_frames > temporal.shape[0]:
        raise ValueError(
            f"RoPE table supports {temporal.shape[0]} frames, requested "
            f"[{absolute_start_frame}, {absolute_start_frame + num_frames})"
        )
    if precision not in {"native", "complex64"}:
        raise ValueError("RoPE precision must be native or complex64")
    work_dtype = torch.float32 if precision == "complex64" else torch.float64
    # The native path follows upstream causal_rope_apply in complex128.  The
    # legacy long cached-K reader intentionally converts each model block to
    # float32/complex64 instead, so legacy mode reproduces that branch.
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
    """Compare the fused kernel with the reference once per geometry.

    The kernel exists to change speed, not values, so the first time it meets
    a geometry it rotates synthetic keys at the first and the last frames of
    the RoPE table both ways and refuses to continue unless every bit agrees.
    """
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
    """Rotate a pre-RoPE cache read, one absolute-position span at a time.

    ``x`` is ``[B, S, H, D]`` holding the spans back to back in the order
    given, which is the order :meth:`ChunkedKVCache.spans` returns.

    ``out`` receives the rotated keys instead of a fresh allocation, and may
    alias ``x``: each block is read in full before its result is written back,
    so rotating a cache read in place is safe and saves a tensor the size of
    the whole resident range on every forward.
    """
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [B,S,H,D]")
    if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
        raise ValueError("grid_sizes must have shape [B,3]")
    if grid_sizes.shape[0] != x.shape[0]:
        raise ValueError(
            f"grid_sizes batch {grid_sizes.shape[0]} does not match input batch "
            f"{x.shape[0]}"
        )
    if torch.any(grid_sizes <= 0):
        raise ValueError("grid sizes must be positive")
    spans = list(spans)
    if not spans:
        raise ValueError("RoPE requires at least one cache span")
    channels = x.size(3) // 2
    temporal, height_freqs, width_freqs = freqs.split(
        [channels - 2 * (channels // 3), channels // 3, channels // 3], dim=1
    )
    if out is None:
        result = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("RoPE output buffer must match the input")
        result = out
    fused = _fused_rotation(x, result, freqs, precision)
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
        if fused is not None:
            if max(height, width) > temporal.shape[0]:
                raise ValueError("RoPE table is smaller than the latent grid")
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
        precision=precision,
        out=out,
    )


__all__ = [
    "ROPE_BACKEND_ENV",
    "apply_rope_by_span",
    "causal_rope_apply_long_input",
    "rope_backend",
]
