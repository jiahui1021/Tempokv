"""Independent, correction-sensitive answers for every current launch candidate.

These cases do not call query_channels or the random-test reference to derive
their answers. All logits/values have closed-form results, including a distant
unquantized key whose output contribution must survive the online KV loop.
"""

import ast
import math
from pathlib import Path

import pytest
import torch


def candidate_names():
    # Collect CPU-side without importing Triton, and follow future tile additions.
    path = Path(__file__).resolve().parents[2] / "src/tempokv/runtime/jensen_triton_kernel.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_TILES" for target in node.targets
        ):
            return ("legacy", *ast.literal_eval(node.value), "auto")
    raise AssertionError("Triton launch candidate table was not found")


CONFIGS = candidate_names()


def correction_strength(dtype):
    # Each active group has exactly 64 unit entries; Delta squared is exactly 48.
    coefficient = torch.tensor(-64 / (24 * math.sqrt(128)), dtype=torch.float32).to(dtype)
    return -float(coefficient) * 48 / math.sqrt(128)


def grouped_case(dtype, device):
    q = torch.zeros(1, 2, 1, 128, dtype=dtype, device=device)
    q[:, 0, :, :64] = 1
    q[:, 1, :, 64:] = 1
    k = torch.zeros(1, 3, 1, 128, dtype=dtype, device=device)
    v = torch.zeros_like(k)
    for token in range(3):
        v[:, token, :, token] = 1
    steps = torch.zeros(1, 3, 1, 2, dtype=dtype, device=device)
    steps[:, 0, :, 0] = 48
    steps[:, 1, :, 1] = 48
    # Key 2 is unquantized. Q0 corrects key 0; Q1 corrects key 1.
    weight = math.exp(-correction_strength(dtype))
    expected = torch.zeros(q.shape, dtype=torch.float32, device=device)
    expected[:, 0, :, :3] = torch.tensor([weight, 1, 1], device=device) / (weight + 2)
    expected[:, 1, :, :3] = torch.tensor([1, weight, 1], device=device) / (weight + 2)
    return q, k, v, steps, expected


def far_tail_case(dtype, device, length):
    q = torch.zeros(1, 1, 1, 128, dtype=dtype, device=device)
    q[..., :64] = 1
    k = torch.zeros(1, length, 1, 128, dtype=dtype, device=device)
    v = torch.zeros_like(k)
    v[:, -1] = 1
    steps = torch.full((1, length, 1, 2), 48, dtype=dtype, device=device)
    steps[:, -1] = 0
    strength = correction_strength(dtype)
    tail_key = torch.tensor((math.log(length - 1) - strength) * math.sqrt(128) / 64,
                            dtype=dtype)
    k[:, -1, :, :64] = float(tail_key)
    tail_logit = float(tail_key) * 64 / math.sqrt(128)
    tail_probability = 1 / (1 + (length - 1) * math.exp(-strength - tail_logit))
    expected = torch.full(q.shape, tail_probability, dtype=torch.float32, device=device)
    return q, k, v, steps, expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_analytic_cases_detect_missing_correction(dtype):
    *_, expected = grouped_case(dtype, "cpu")
    assert abs(float(expected[0, 0, 0, 0]) - 1 / 3) > 0.15
    for length in (4097, 65537):
        _, k, _, _, tail = far_tail_case(dtype, "cpu", length)
        uncorrected_logit = float(k[0, -1, 0, 0]) * 64 / math.sqrt(128)
        uncorrected = 1 / (1 + (length - 1) * math.exp(-uncorrected_logit))
        assert 0.48 < float(tail.flatten()[0]) < 0.52
        assert abs(float(tail.flatten()[0]) - uncorrected) > 0.20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA/Triton")
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("length", [3, 4097, 65537])
@torch.no_grad()
def test_all_candidates_match_independent_analytic_answer(config, dtype, length):
    from tempokv.runtime.jensen_triton_kernel import attention_forward

    q, k, v, steps, expected = (
        grouped_case(dtype, "cuda") if length == 3 else far_tail_case(dtype, "cuda", length)
    )
    actual = attention_forward(q, k, v, steps, 64, config=config)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual.float(), expected, atol=0.003, rtol=0.005)
