# References:
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/rotary_positional_embedding.py

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only development and test environments.
    triton = None
    tl = None


# All 48 LongCat attention layers use the same RoPE geometry.  Retain one
# device copy per process/device instead of copying the frequency tensor and
# rebuilding cos/sin in every layer and every denoising step.  The entry is
# deliberately bounded to one geometry per device; a new resolution replaces
# the old entry so the cache cannot accumulate video-sized tensors.
_DEVICE_TRIG_CACHE: dict[
    tuple[str, int | None],
    tuple[tuple[object, ...], int, torch.Tensor, torch.Tensor],
] = {}


def clear_shared_rope_device_cache() -> None:
    """Release the shared device table without synchronizing the accelerator."""
    _DEVICE_TRIG_CACHE.clear()


def clear_shared_rope_cache() -> None:
    """Release shared CPU/device RoPE caches, primarily for model teardown."""
    clear_shared_rope_device_cache()
    _precompute_freqs_cis_3d.cache_clear()


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    original_shape = x.shape
    pairs = x.reshape(*original_shape[:-1], original_shape[-1] // 2, 2)
    x1, x2 = pairs.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(original_shape)


def _normalize_grid_size(grid_size) -> tuple[int, int, int]:
    if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 3:
        raise ValueError("LongCat RoPE grid_size must contain (frames, height, width)")
    result = tuple(int(value) for value in grid_size)
    if any(value <= 0 for value in result):
        raise ValueError("LongCat RoPE grid dimensions must be positive")
    return result


def _normalize_cp_split(cp_split_hw) -> tuple[int, int]:
    if cp_split_hw is None:
        return (1, 1)
    if not isinstance(cp_split_hw, (tuple, list)) or len(cp_split_hw) != 2:
        raise ValueError("LongCat cp_split_hw must contain (height, width)")
    result = tuple(int(value) for value in cp_split_hw)
    if any(value <= 0 for value in result):
        raise ValueError("LongCat cp_split_hw values must be positive")
    return result


def _context_parallel_rank(cp_split_hw: tuple[int, int]) -> int:
    cp_size = cp_split_hw[0] * cp_split_hw[1]
    if cp_size == 1:
        return 0
    # Keep the heavyweight distributed helpers lazy so CPU-only users can
    # import and test RoPE without installing all LongCat optional packages.
    from ..context_parallel import context_parallel_util

    rank = context_parallel_util.get_cp_rank()
    if rank is None:
        raise RuntimeError("LongCat context parallel RoPE requires an initialized CP rank")
    rank = int(rank)
    if not 0 <= rank < cp_size:
        raise ValueError(f"LongCat CP rank {rank} is outside [0, {cp_size})")
    return rank


@lru_cache(maxsize=8)
def _precompute_freqs_cis_3d(
    head_dim: int,
    base: int,
    grid_size: tuple[int, int, int],
    cp_split_hw: tuple[int, int],
    cp_rank: int,
) -> torch.Tensor:
    """Build one CPU frequency table shared by every transformer layer."""
    num_frames, height, width = grid_size
    split_h, split_w = cp_split_hw
    if height % split_h or width % split_w:
        raise ValueError(
            f"LongCat RoPE grid {(height, width)} is not divisible by "
            f"CP split {cp_split_hw}"
        )

    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    freqs_t = 1.0 / (
        base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t)
    )
    freqs_h = 1.0 / (
        base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h)
    )
    freqs_w = 1.0 / (
        base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w)
    )
    grid_t = torch.from_numpy(
        np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
    ).float()
    grid_h = torch.from_numpy(
        np.linspace(0, height, height, endpoint=False, dtype=np.float32)
    ).float()
    grid_w = torch.from_numpy(
        np.linspace(0, width, width, endpoint=False, dtype=np.float32)
    ).float()
    freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
    freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
    freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
    freqs_t = freqs_t.repeat_interleave(2, dim=-1)
    freqs_h = freqs_h.repeat_interleave(2, dim=-1)
    freqs_w = freqs_w.repeat_interleave(2, dim=-1)
    freqs = broadcat(
        (
            freqs_t[:, None, None, :],
            freqs_h[None, :, None, :],
            freqs_w[None, None, :, :],
        ),
        dim=-1,
    )

    if split_h * split_w > 1:
        local_height = height // split_h
        local_width = width // split_w
        height_rank = cp_rank // split_w
        width_rank = cp_rank % split_w
        height_start = height_rank * local_height
        width_start = width_rank * local_width
        freqs = freqs[
            :,
            height_start : height_start + local_height,
            width_start : width_start + local_width,
            :,
        ].contiguous()
    return freqs.reshape(-1, head_dim)


if triton is not None:

    @triton.jit
    def _triton_rope_kernel(
        input_ptr,
        input_stride_b,
        input_stride_h,
        input_stride_s,
        input_stride_d,
        output_ptr,
        output_stride_b,
        output_stride_h,
        output_stride_s,
        output_stride_d,
        cos_ptr,
        cos_stride_s,
        cos_stride_d,
        sin_ptr,
        sin_stride_s,
        sin_stride_d,
        heads: tl.constexpr,
        sequence: tl.constexpr,
        width: tl.constexpr,
        position_offset: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        batch = batch_head // heads
        head = batch_head - batch * heads
        token = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
        pair = tl.arange(0, BLOCK_PAIRS)
        even_channel = pair * 2
        odd_channel = even_channel + 1
        valid = (token[:, None] < sequence) & (odd_channel[None, :] < width)

        input_base = batch * input_stride_b + head * input_stride_h
        even_input_offset = (
            input_base
            + token[:, None] * input_stride_s
            + even_channel[None, :] * input_stride_d
        )
        odd_input_offset = (
            input_base
            + token[:, None] * input_stride_s
            + odd_channel[None, :] * input_stride_d
        )
        x_even = tl.load(input_ptr + even_input_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        x_odd = tl.load(input_ptr + odd_input_offset, mask=valid, other=0.0).to(
            tl.float32
        )

        position = token + position_offset
        trig_offset = (
            position[:, None] * cos_stride_s
            + even_channel[None, :] * cos_stride_d
        )
        cos = tl.load(cos_ptr + trig_offset, mask=valid, other=0.0).to(tl.float32)
        sin_offset = (
            position[:, None] * sin_stride_s
            + even_channel[None, :] * sin_stride_d
        )
        sin = tl.load(sin_ptr + sin_offset, mask=valid, other=0.0).to(tl.float32)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        output_base = batch * output_stride_b + head * output_stride_h
        even_output_offset = (
            output_base
            + token[:, None] * output_stride_s
            + even_channel[None, :] * output_stride_d
        )
        odd_output_offset = (
            output_base
            + token[:, None] * output_stride_s
            + odd_channel[None, :] * output_stride_d
        )
        tl.store(output_ptr + even_output_offset, out_even, mask=valid)
        tl.store(output_ptr + odd_output_offset, out_odd, mask=valid)


def _triton_rotate(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    batch, heads, sequence, width = tensor.shape
    if output is None:
        output = torch.empty_like(tensor, memory_format=torch.contiguous_format)
    elif (
        output.shape != tensor.shape
        or output.dtype != tensor.dtype
        or output.device != tensor.device
    ):
        raise ValueError("Triton RoPE output must match the input tensor")
    # D=128 already contributes 64 rotary pairs.  Sixteen tokens keep the
    # per-program tile at 1024 pairs, avoiding the register pressure and
    # compilation risk of the 4096-pair legacy reference tile.
    block_s = 16
    block_pairs = triton.next_power_of_2(width // 2)
    grid = (batch * heads, triton.cdiv(sequence, block_s))
    _triton_rope_kernel[grid](
        tensor,
        tensor.stride(0),
        tensor.stride(1),
        tensor.stride(2),
        tensor.stride(3),
        output,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        cos,
        cos.stride(0),
        cos.stride(1),
        sin,
        sin.stride(0),
        sin.stride(1),
        heads=heads,
        sequence=sequence,
        width=width,
        position_offset=position_offset,
        BLOCK_S=block_s,
        BLOCK_PAIRS=block_pairs,
        num_warps=8 if block_pairs >= 64 else 4,
    )
    return output


def _torch_rotate(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int,
) -> torch.Tensor:
    sequence = int(tensor.shape[2])
    cos = cos[position_offset : position_offset + sequence].view(
        1, 1, sequence, tensor.shape[-1]
    )
    sin = sin[position_offset : position_offset + sequence].view(
        1, 1, sequence, tensor.shape[-1]
    )
    tensor_float = tensor.float()
    output = (tensor_float * cos) + (rotate_half(tensor_float) * sin)
    return output.type_as(tensor)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, head_dim, cp_split_hw=None):
        """Rotary positional embedding for LongCat's flattened 3D grid."""
        super().__init__()
        self.head_dim = int(head_dim)
        assert self.head_dim % 8 == 0, "Dim must be a multiply of 8 for 3D RoPE."
        self.cp_split_hw = _normalize_cp_split(cp_split_hw)
        self.base = 10000
        # Keep the public per-instance mapping for compatibility.  Values are
        # references to the process-wide lru_cache and do not duplicate data.
        self.freqs_dict = {}

    def register_grid_size(self, grid_size):
        grid_size = _normalize_grid_size(grid_size)
        cp_rank = _context_parallel_rank(self.cp_split_hw)
        if grid_size not in self.freqs_dict:
            self.freqs_dict[grid_size] = _precompute_freqs_cis_3d(
                self.head_dim,
                self.base,
                grid_size,
                self.cp_split_hw,
                cp_rank,
            )

    def precompute_freqs_cis_3d(self, grid_size):
        grid_size = _normalize_grid_size(grid_size)
        return _precompute_freqs_cis_3d(
            self.head_dim,
            self.base,
            grid_size,
            self.cp_split_hw,
            _context_parallel_rank(self.cp_split_hw),
        )

    def _device_trig(
        self, device: torch.device, grid_size: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.register_grid_size(grid_size)
        cp_rank = _context_parallel_rank(self.cp_split_hw)
        frames, height, width = grid_size
        signature = (
            self.head_dim,
            self.base,
            height,
            width,
            self.cp_split_hw,
            cp_rank,
        )
        device_key = (device.type, device.index)
        cached = _DEVICE_TRIG_CACHE.get(device_key)
        local_sequence = int(self.freqs_dict[grid_size].shape[0])
        if cached is not None:
            cached_signature, cached_frames, cos, sin = cached
            if cached_signature == signature and cached_frames >= frames:
                return cos[:local_sequence], sin[:local_sequence]
            _DEVICE_TRIG_CACHE.pop(device_key)
            del cached

        freqs = self.freqs_dict[grid_size].to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        cos = freqs.cos()
        sin = freqs.sin()
        del freqs
        _DEVICE_TRIG_CACHE[device_key] = (signature, frames, cos, sin)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size,
        *,
        q_position_offset: int = 0,
        k_position_offset: int = 0,
        inplace_key: bool = False,
    ):
        """Apply 3D RoPE to differently sized query and key sequences.

        Position offsets index the local, flattened ``T*H*W`` sequence after
        context-parallel spatial sharding.  Cached attention can therefore
        rotate only the current query tokens instead of allocating a cache-sized
        dummy query prefix.
        """
        grid_size = _normalize_grid_size(grid_size)
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("LongCat RoPE expects Q/K in [B,H,S,D] layout")
        if q.device != k.device:
            raise ValueError("LongCat RoPE Q/K must be on the same device")
        if q.dtype != k.dtype:
            raise ValueError("LongCat RoPE Q/K must use the same dtype")
        if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
            raise ValueError("LongCat RoPE Q/K batch and head dimensions must match")
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"LongCat RoPE expected head_dim={self.head_dim}, got "
                f"Q={q.shape[-1]}, K={k.shape[-1]}"
            )
        for name, value in (
            ("q_position_offset", q_position_offset),
            ("k_position_offset", k_position_offset),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(inplace_key, bool):
            raise ValueError("inplace_key must be a boolean")

        cos, sin = self._device_trig(q.device, grid_size)
        q_end = q_position_offset + int(q.shape[2])
        k_end = k_position_offset + int(k.shape[2])
        if q_end > cos.shape[0] or k_end > cos.shape[0]:
            raise ValueError(
                "LongCat RoPE position range exceeds the context-parallel grid: "
                f"Q end={q_end}, K end={k_end}, grid tokens={cos.shape[0]}"
            )

        use_triton = (
            triton is not None
            and q.is_cuda
            and k.is_cuda
            and q.dtype == torch.bfloat16
            and (not torch.is_grad_enabled() or (not q.requires_grad and not k.requires_grad))
        )
        if use_triton:
            q_output = _triton_rotate(q, cos, sin, q_position_offset)
            # Cached attention has already consumed the pre-RoPE K as the
            # CrossKV source.  Rotating that contiguous final buffer in place
            # removes another full-sequence K allocation.
            k_output = _triton_rotate(
                k,
                cos,
                sin,
                k_position_offset,
                output=k if inplace_key else None,
            )
        else:
            q_output = _torch_rotate(q, cos, sin, q_position_offset)
            k_output = _torch_rotate(k, cos, sin, k_position_offset)
        return q_output, k_output
