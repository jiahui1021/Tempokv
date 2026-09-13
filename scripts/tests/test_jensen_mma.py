"""New Tensor Core correction versus rounded reference and analytic answers."""

import pytest
import torch

from tempokv.runtime.jensen import query_channels
from test_jensen_triton import reference
from test_jensen_triton_analytic import far_tail_case, grouped_case


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rank16_padding_preserves_rounded_bilinear_correction(group_size, dtype):
    torch.manual_seed(71)
    q = torch.randn(2, 17, 3, 128, dtype=dtype)
    steps = torch.rand(2, 65, 3, 128 // group_size).to(dtype).square()
    steps[:, -7:] = 0
    coeff = query_channels(q, group_size).to(dtype)
    q16 = torch.zeros(2, 3, 17, 16, dtype=dtype)
    k16 = torch.zeros(2, 3, 65, 16, dtype=dtype)
    q16[..., :128 // group_size] = coeff.permute(0, 2, 1, 3)
    k16[..., :128 // group_size] = steps.permute(0, 2, 1, 3)
    expected = torch.einsum("bqhg,bkhg->bhqk", coeff.float(), steps.float())
    actual = q16.float() @ k16.float().transpose(-1, -2)
    torch.testing.assert_close(actual, expected)
    assert not actual[..., -7:].any()


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA/Triton")
CONFIGS = ("64x64w8s3", "128x64w8s4")


@cuda_only
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim,group_size", [(64, 32), (64, 64), (128, 32), (128, 64), (128, 128)])
@torch.no_grad()
def test_mma_matches_reference_for_strided_groups(config, dtype, dim, group_size):
    from tempokv.runtime.jensen_triton_kernel import attention_forward

    torch.manual_seed(731)
    q = torch.randn(2, 130, 4, dim * 2, dtype=dtype, device="cuda")[:, ::2, ::2, ::2]
    k = torch.randn(2, 514, 4, dim * 2, dtype=dtype, device="cuda")[:, ::2, ::2, ::2]
    v = torch.randn_like(k)
    steps = torch.rand(2, 514, 4, 2 * (dim // group_size), dtype=dtype, device="cuda")[:, ::2, ::2, ::2]
    steps.mul_(2).square_()
    steps[:, -17:] = 0
    for multiplier in (1, 0):
        steps.mul_(multiplier)
        expected = reference(q, k, v, steps, group_size)
        actual = attention_forward(q, k, v, steps, group_size, config=config, correction_mode="mma")
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@cuda_only
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("length", [3, 4097])
@torch.no_grad()
def test_mma_matches_analytic_correction_and_tail(config, dtype, length):
    from tempokv.runtime.jensen_triton_kernel import attention_forward

    q, k, v, steps, expected = (grouped_case(dtype, "cuda") if length == 3 else
                               far_tail_case(dtype, "cuda", length))
    actual = attention_forward(q, k, v, steps, 64, config=config, correction_mode="mma")
    torch.testing.assert_close(actual.float(), expected, atol=0.003, rtol=0.005)
