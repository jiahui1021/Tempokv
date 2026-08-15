from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from tempokv.config import CodecConfig
from tempokv.parameters import CrossKVParameters, TemporalParameters


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_v_predictors.py"
SPEC = importlib.util.spec_from_file_location("compare_v_predictors", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_candidate_accepts_trace_array_field_names() -> None:
    parameters = inspect.signature(MODULE._candidate).parameters
    assert {"keys", "values", "q_post_rope", "k_post_rope"} <= set(parameters)


def test_ratio_summary_preserves_prompt_layer_and_head_axes() -> None:
    reference = torch.ones(2, 4, 3, 2, 1)
    error = reference.clone()
    error[:, :2] *= 0.5
    report = MODULE.ratio_summary(error, reference, prompt_count=2)

    assert report["pooled_nmse"] == pytest.approx(0.625)
    assert report["per_prompt_nmse"] == pytest.approx([0.25, 1.0])
    assert report["per_layer_nmse"] == pytest.approx([0.625, 0.625])
    assert report["per_head_nmse"] == pytest.approx([0.625, 0.625])


def test_prediction_helpers_match_simple_closed_loop_formulas() -> None:
    target = torch.tensor([[[[[1.0]], [[2.0]], [[4.0]]]]])
    rebuilt = torch.tensor([[[[[1.0]], [[1.5]], [[3.0]]]]])
    temporal = TemporalParameters(
        history_weights=torch.tensor([[[0.5]]]),
        history_bias=torch.tensor([[0.25]]),
    )
    temporal_prediction = MODULE.temporal_predictions(target, rebuilt, temporal)
    assert temporal_prediction.flatten().tolist() == pytest.approx([0.0, 0.75, 1.0])

    cross = CrossKVParameters(
        weight=torch.tensor([[[2.0]]]),
        bias=torch.tensor([[0.25]]),
        history_weights=torch.tensor([[[0.5]]]),
    )
    config = CodecConfig(
        k_bits=4,
        v_bits=4,
        anchor_bits=4,
        group_size=1,
        stride=1,
        k_history_length=1,
        v_history_length=1,
        scale_dtype="float32",
        reconstruction_dtype="float32",
        use_triton=False,
    )
    source = torch.tensor([[[[[1.0]], [[2.0]], [[3.0]]]]])
    cross_prediction = MODULE.cross_kv_predictions(source, rebuilt, cross, config)
    assert cross_prediction.flatten().tolist() == pytest.approx([0.0, 4.75, 7.0])
