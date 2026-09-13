"""Inference-only Jensen attention with no widened heads or score buffers.

The CUDA implementation is imported lazily so CPU users need not have Triton.
Inputs are BSHD, steps are BSHG and already squared in the cache dtype. This
entry point only supports the unmasked, equal-head Self-Forcing cache window.
"""

from __future__ import annotations

import torch


def corrected_attention(query, key, value, step_squared, *, group_size):
    """Apply the existing rounded Jensen correction in a single warmed GPU launch.

    Supported head dimensions: 64/128; group sizes: 32/64/128 dividing D.
    Arbitrary positive input strides are accepted, including sliced windows.
    This is forward-only; callers requesting gradients receive an error.
    The default auto configuration benchmarks candidate tiles on a new length
    bucket; use TEMPOKV_JENSEN_TRITON_CONFIG for a measured fixed configuration.
    TEMPOKV_JENSEN_TRITON_CORRECTION=mma opts into the small Tensor Core
    correction; scalar remains the user-measured default.
    TEMPOKV_JENSEN_TRITON_IMPL=pipeline selects an experimental schedule;
    it requires MMA correction and a fixed config. Baseline stays the default.
    """
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Q/K/V must be rank-4 BSHD tensors, with matching K/V")
    if min(*query.shape, *key.shape) <= 0:
        raise ValueError("Q/K dimensions must be non-empty")
    if (query.shape[0], query.shape[2], query.shape[3]) != (
        key.shape[0], key.shape[2], key.shape[3]
    ):
        raise ValueError("Q/K must have matching batch, heads and head dimension")
    dim = query.shape[-1]
    if dim not in (64, 128):
        raise ValueError("Triton Jensen requires head dimension 64 or 128")
    if group_size not in (32, 64, 128) or dim % group_size:
        raise ValueError("group_size must be 32, 64 or 128 and divide the head dimension")
    if step_squared.shape != (*key.shape[:-1], dim // group_size):
        raise ValueError("squared steps must align token-for-token with K")
    tensors = (query, key, value, step_squared)
    if any(t.device != query.device for t in tensors):
        raise ValueError("Q/K/V and squared steps must share a device")
    if query.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != query.dtype for t in tensors
    ):
        raise ValueError("Triton Jensen inputs must share a BF16 or FP16 dtype")
    if any(s <= 0 for t in tensors for s in t.stride()):
        raise ValueError("Triton Jensen inputs require positive strides")
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("Triton Jensen is inference-only; gradients are unsupported")
    if query.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("Triton Jensen requires NVIDIA CUDA")
    with torch.cuda.device(query.device):
        if torch.cuda.get_device_capability()[0] < 8:
            raise RuntimeError("Triton Jensen requires an Ampere or newer GPU")
        from .jensen_triton_kernel import attention_forward

        return attention_forward(query, key, value, step_squared, group_size)


__all__ = ["corrected_attention"]
