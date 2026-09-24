"""Inference-only Jensen attention without widened heads or score buffers."""

from __future__ import annotations

import torch


def corrected_attention(query, key, value, step_squared, *, group_size):
    """Apply the Jensen correction in one GPU launch (forward only).

    Head dim 64/128, group size 32/64/128.  ``TEMPOKV_JENSEN_TRITON_CONFIG``,
    ``..._CORRECTION`` and ``..._IMPL`` select tiles, correction and schedule.
    """
    dim = query.shape[-1]
    if dim not in (64, 128):
        raise ValueError("Triton Jensen requires head dimension 64 or 128")
    if group_size not in (32, 64, 128) or dim % group_size:
        raise ValueError("group_size must be 32, 64 or 128 and divide the head dimension")
    if query.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("Triton Jensen requires NVIDIA CUDA")
    with torch.cuda.device(query.device):
        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("Triton Jensen requires an Ampere or newer GPU")
        from .jensen_triton_kernel import attention_forward

        return attention_forward(query, key, value, step_squared, group_size)


__all__ = ["corrected_attention"]
