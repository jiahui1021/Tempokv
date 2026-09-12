"""CUDA reference tests for the actual custom Jensen kernel, plus CPU guards."""

import pytest
import torch

from tempokv.runtime.jensen import query_channels
from tempokv.runtime.jensen_triton import corrected_attention


def test_guards_and_lazy_cuda_import():
    q = torch.zeros(1, 3, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 5, 2, 128, dtype=q.dtype)
    steps = torch.zeros(1, 5, 2, 2, dtype=q.dtype)
    with pytest.raises(ValueError, match="group_size"):
        corrected_attention(q, k, k, steps, group_size=16)
    with pytest.raises(ValueError, match="token-for-token"):
        corrected_attention(q, k, k, steps[:, :2], group_size=64)
    with pytest.raises(ValueError, match="dtype"):
        corrected_attention(q, k, k, steps.float(), group_size=64)
    with pytest.raises(ValueError, match="positive strides"):
        corrected_attention(q.expand(2, -1, -1, -1), k.expand(2, -1, -1, -1),
                            k.expand(2, -1, -1, -1), steps.expand(2, -1, -1, -1), group_size=64)
    with pytest.raises(RuntimeError, match="inference-only"):
        corrected_attention(q.requires_grad_(), k, k, steps, group_size=64)
    with torch.no_grad(), pytest.raises(RuntimeError, match="CUDA"):
        corrected_attention(q, k, k, steps, group_size=64)


def reference(q, k, v, steps, group_size):
    dim = q.shape[-1]
    qt = query_channels(q, group_size).to(q.dtype).float().permute(0, 2, 1, 3)
    kt = steps.float().permute(0, 2, 1, 3)
    scores = q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1)
    scores = (scores + qt @ kt.transpose(-1, -2)) * dim ** -0.5
    return (scores.softmax(-1) @ v.float().permute(0, 2, 1, 3)).permute(0, 2, 1, 3)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA/Triton")


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim,group_size", [(64, 32), (64, 64), (128, 32), (128, 64), (128, 128)])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_cuda_matches_reference_with_growing_cache(dtype, dim, group_size, strided):
    torch.manual_seed(651)
    for nq, nk in ((1, 1), (17, 63), (65, 129), (128, 256), (65, 129)):
        # Sliced B/H/D and group axes catch hidden contiguous-layout assumptions.
        def tensor(length, width):
            if strided:
                return torch.randn(4, length * 2, 4, width * 2, device="cuda", dtype=dtype)[::2, ::2, ::2, ::2]
            return torch.randn(2, length, 2, width, device="cuda", dtype=dtype)

        q, k, v = tensor(nq, dim), tensor(nk, dim), tensor(nk, dim)
        steps = tensor(nk, dim // group_size)
        steps.abs_().clamp_(max=2).square_()
        steps[:, -min(17, nk):] = 0  # unquantized current chunk
        expected = reference(q, k, v, steps, group_size)
        actual = corrected_attention(q, k, v, steps, group_size=group_size)
        assert actual.shape == q.shape and actual.is_contiguous()
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@cuda_only
@torch.no_grad()
def test_zero_steps_large_logits_and_fresh_steps():
    torch.manual_seed(811)
    q = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16) * 8
    k = torch.randn(1, 513, 2, 128, device="cuda", dtype=q.dtype) * 8
    v = torch.randn_like(k)
    steps = torch.zeros(1, 513, 2, 2, device="cuda", dtype=q.dtype)
    for multiplier in (0.0, 2.0, 0.0):
        steps[:, :256] = multiplier
        actual = corrected_attention(q, k, v, steps, group_size=64)
        assert actual.isfinite().all()
        torch.testing.assert_close(actual.float(), reference(q, k, v, steps, 64), atol=0.03, rtol=0.03)
