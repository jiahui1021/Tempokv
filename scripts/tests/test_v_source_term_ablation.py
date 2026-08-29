from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_v_source_terms.py"
SPEC = importlib.util.spec_from_file_location("compare_v_source_terms", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_candidate_term_sets_are_exactly_three_singles_and_three_pairs() -> None:
    assert MODULE.TERM_SETS == {
        "cross": ("cross",),
        "temporal": ("temporal",),
        "innovation": ("innovation",),
        "cross_temporal": ("cross", "temporal"),
        "cross_innovation": ("cross", "innovation"),
        "temporal_innovation": ("temporal", "innovation"),
    }


def test_clean_and_reconstructed_key_innovation_are_explicit() -> None:
    clean = torch.tensor([[[[[10.0]], [[20.0]]]]])
    rebuilt = torch.tensor([[[[[9.0]], [[18.0]]]]])
    prediction = torch.tensor([[[[[0.0]], [[12.0]]]]])
    clean_result = MODULE.key_innovation(
        clean, rebuilt, prediction, source="clean"
    )
    deployable_result = MODULE.key_innovation(
        clean, rebuilt, prediction, source="reconstructed"
    )
    assert clean_result.flatten().tolist() == pytest.approx([0.0, 8.0])
    assert deployable_result.flatten().tolist() == pytest.approx([0.0, 6.0])


def _parameters(terms: tuple[str, ...]) -> object:
    values = {"cross": 2.0, "temporal": 3.0, "innovation": 5.0}
    return MODULE.JointVParameters(
        terms=terms,
        weights={term: torch.tensor([[[[values[term]]]]]) for term in terms},
        bias=torch.tensor([[[7.0]]]),
        weight_mode="layer_head",
    )


@pytest.mark.parametrize(
    ("terms", "expected"),
    [
        (("cross",), 9.0),
        (("temporal",), 10.0),
        (("innovation",), 12.0),
        (("cross", "temporal"), 12.0),
        (("cross", "innovation"), 14.0),
        (("temporal", "innovation"), 15.0),
    ],
)
def test_terms_and_pairs_add_one_bias(terms: tuple[str, ...], expected: float) -> None:
    sources = {term: torch.ones(1, 1, 1) for term in terms}
    result = MODULE.predict_terms(_parameters(terms), sources)
    assert result.item() == pytest.approx(expected)


def test_temporal_source_reads_previous_reconstruction() -> None:
    rebuilt = torch.tensor([[[[[2.0]], [[4.0]], [[8.0]]]]])
    source = MODULE.temporal_source(rebuilt)
    assert source.flatten().tolist() == pytest.approx([0.0, 2.0, 4.0])


def test_joint_fit_refits_selected_sources_with_one_bias() -> None:
    cross = torch.arange(1.0, 13.0).reshape(1, 2, 3, 1, 2)
    innovation = torch.flip(cross, dims=(-1,)) * 0.25
    target = 2.0 * cross - 3.0 * innovation + 1.5
    fitted = MODULE.fit_joint_dense_predictor(
        ("cross", "innovation"),
        {"cross": cross, "innovation": innovation},
        target,
        ridge=1e-6,
    )
    predicted = MODULE.predict_terms(
        fitted,
        {"cross": cross[:, :, 1], "innovation": innovation[:, :, 1]},
    )
    assert torch.allclose(predicted, target[:, :, 1], atol=2e-3, rtol=2e-3)
    assert fitted.parameter_count == 10
    assert fitted.weight_mode == "layer_shared"


def test_layer_shared_fit_has_one_map_per_head_not_per_layer() -> None:
    source = torch.randn(3, 4, 3, 2, 5)
    target = source * 0.5 + 0.25
    fitted = MODULE.fit_joint_dense_predictor(
        ("cross",), {"cross": source}, target, ridge=0.1
    )
    assert fitted.weights["cross"].shape == (2, 5, 5)
    assert fitted.bias.shape == (2, 5)
    assert fitted.parameter_count == 60


def test_vector_metrics_report_reconstruction_error_and_cosine() -> None:
    clean = torch.tensor([[[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]]])
    same = MODULE.vector_metrics(clean, clean)
    assert same["pooled_mse"] == 0.0
    assert same["pooled_nmse"] == 0.0
    assert same["pooled_cosine_similarity"] == pytest.approx(1.0)

    orthogonal = clean.clone()
    orthogonal[..., 1:, :, :] = torch.tensor([0.0, 1.0])
    orthogonal_metrics = MODULE.vector_metrics(clean, orthogonal)
    assert orthogonal_metrics["pooled_cosine_similarity"] == pytest.approx(0.0)

    negative = clean.clone()
    negative[..., 1:, :, :] *= -1
    negative_metrics = MODULE.vector_metrics(clean, negative)
    assert negative_metrics["pooled_cosine_similarity"] == pytest.approx(-1.0)

    zeros = torch.zeros_like(clean)
    zero_metrics = MODULE.vector_metrics(zeros, zeros)
    assert zero_metrics["pooled_cosine_similarity"] == 0.0
    assert all(
        torch.isfinite(torch.tensor(zero_metrics[name]))
        for name in ("pooled_mse", "pooled_nmse", "pooled_cosine_similarity")
    )
