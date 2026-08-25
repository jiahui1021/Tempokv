from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "experiment_crosskv_compression.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("crosskv_compression_experiment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def test_quantized_candidates_have_expected_storage_and_finite_weights() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(2, 3, 8, 8, generator=generator)
    candidates = {
        item.name: item
        for item in experiment.candidate_stream(
            weight, methods=("fp16", "int8", "int4"), ranks=(2,)
        )
    }

    assert candidates["dense_fp32"].stored_weight_bytes == weight.numel() * 4
    assert candidates["fp16"].stored_weight_bytes == weight.numel() * 2
    scale_count = 2 * 3 * 8
    assert candidates["int8"].stored_weight_bytes == weight.numel() + 2 * scale_count
    assert candidates["int4"].stored_weight_bytes == weight.numel() // 2 + 2 * scale_count
    assert torch.isfinite(candidates["int4"].weight).all()


def test_per_matrix_svd_recovers_an_exact_low_rank_bank() -> None:
    generator = torch.Generator().manual_seed(11)
    left = torch.randn(2, 3, 8, 2, generator=generator)
    right = torch.randn(2, 3, 2, 8, generator=generator)
    weight = torch.matmul(left, right)
    candidates = list(
        experiment.candidate_stream(weight, methods=("svd",), ranks=(1, 2, 3))
    )
    rank_two = next(item for item in candidates if item.name == "svd_r2")

    torch.testing.assert_close(rank_two.weight, weight, rtol=2e-5, atol=2e-5)
    assert rank_two.stored_scalars == 2 * 3 * 2 * 8 * 2
    assert rank_two.stored_weight_bytes < weight.numel() * 4


def test_shared_tucker_storage_projection_matches_model_shapes() -> None:
    weight = torch.eye(8).repeat(2, 3, 1, 1)
    candidate = next(
        item
        for item in experiment.candidate_stream(
            weight, methods=("shared_tucker",), ranks=(2,)
        )
        if item.family == "shared_tucker"
    )
    projected = experiment._projected_storage(candidate)

    assert projected["self_forcing"]["dense_weight_mib"] == pytest.approx(22.5)
    assert projected["longcat"]["dense_weight_mib"] == pytest.approx(96.0)
    assert projected["self_forcing"]["weight_compression_ratio"] > 1.0


def test_split_matches_existing_clean_trace_protocol() -> None:
    train_ids, eval_ids = experiment.split_trace_ids(104, 5)
    assert eval_ids.tolist() == list(range(4, 104, 5))
    assert len(train_ids) == 84
    assert len(eval_ids) == 20
