"""CUDA checks for the optional attention schedule, including the single tail."""

import pytest
import torch

from test_jensen_triton import reference
from test_jensen_triton_analytic import far_tail_case, grouped_case


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA/Triton")


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim,group_size", [(64, 32), (64, 64), (128, 32), (128, 64), (128, 128)])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_pipeline_reference_at_full_and_partial_tile_boundaries(dtype, dim, group_size, strided):
    from tempokv.runtime.jensen_triton_kernel import attention_forward

    torch.manual_seed(951)

    def tensor(length, width):
        if strided:
            return torch.randn(4, length * 2, 4, width * 2, dtype=dtype, device="cuda")[::2, ::2, ::2, ::2]
        return torch.randn(2, length, 2, width, dtype=dtype, device="cuda")

    q = tensor(65, dim)
    for nk in (1, 32, 64, 65, 128, 129, 257):
        k, v = tensor(nk, dim), tensor(nk, dim)
        steps = tensor(nk, dim // group_size)
        steps.abs_().clamp_(max=2).square_()
        steps[:, -min(nk, 17):] = 0
        for corrected in (True, False):
            if not corrected:
                steps.fill_(float("nan"))  # plain path must not load these values
            expected = reference(q, k, v, steps if corrected else torch.zeros_like(steps), group_size)
            actual = attention_forward(
                q, k, v, steps, group_size, config="128x64w8s4",
                correction_mode="mma", implementation="pipeline", apply_correction=corrected)
            assert actual.is_contiguous() and actual.isfinite().all()
            torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("length", [3, 4097, 65537])
@torch.no_grad()
def test_pipeline_independent_analytic_and_distant_tail(dtype, length):
    from tempokv.runtime.jensen_triton_kernel import attention_forward

    q, k, v, steps, expected = (grouped_case(dtype, "cuda") if length == 3 else
                               far_tail_case(dtype, "cuda", length))
    actual = attention_forward(q, k, v, steps, 64, config="128x64w8s4",
                               correction_mode="mma", implementation="pipeline")
    torch.testing.assert_close(actual.float(), expected, atol=0.003, rtol=0.005)
