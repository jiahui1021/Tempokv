from __future__ import annotations

import itertools
import math

import pytest
import torch

from tempokv.research.precision import (
    METADATA_GROUP_DESCRIPTOR_BYTES,
    METADATA_HEADER_BYTES,
    METADATA_SHAPE_DIM_BYTES,
    PrecisionOption,
    allocate,
    allocation_frontier,
    fake_quantize,
    precision_policy_rows,
)


def test_exact_packing_scales_metadata_and_float_simulation_storage():
    x = torch.tensor([[1., .6, 100., 60., 0.], [2., 0., -3., 0., 5.]])
    result = fake_quantize(x, 3, group_size=4)
    assert result.payload_bytes == math.ceil(10 * 3 / 8) == 4
    assert result.scale_bytes == 2 * math.ceil(5 / 4) * 4 == 16
    assert result.metadata_bytes == METADATA_HEADER_BYTES + 2 * METADATA_SHAPE_DIM_BYTES + METADATA_GROUP_DESCRIPTOR_BYTES == 48
    assert result.logical_bytes == 68
    assert result.details["scales"].shape == (2, 2)
    assert result.details["scale_count"] == 4
    assert result.details["scales"].dtype == torch.float32
    assert result.details["metadata_layout"] == {"header_bytes": 16, "shape_bytes": 16, "group_descriptor_bytes": 16}
    assert result.details["packed_storage_allocated"] is False
    assert result.physical_bytes == x.numel() * x.element_size() + result.scale_bytes == 56
    assert result.physical_bytes == result.details["physical_value_bytes"] + result.details["physical_scale_bytes"]


def test_actual_simulation_cannot_claim_logical_packed_memory_savings():
    x = torch.linspace(-2, 2, 4096).reshape(32, 128)
    result = fake_quantize(x, 2)
    assert result.logical_bytes < x.numel() * x.element_size()
    assert result.physical_bytes >= x.numel() * x.element_size()
    assert result.value.dtype == x.dtype and result.value.device == x.device


def test_independent_last_axis_groups_and_ties_to_even():
    grouped = fake_quantize(torch.tensor([[1., .6, 100., 60., 0.]]), 2, 2)
    torch.testing.assert_close(grouped.value, torch.tensor([[1., 1., 100., 100., 0.]]), atol=0, rtol=0)
    torch.testing.assert_close(grouped.details["scales"], torch.tensor([[1., 100., 1.]]), atol=0, rtol=0)
    rounded = fake_quantize(torch.tensor([-1., -.5, .5, 1.]), 2)
    torch.testing.assert_close(rounded.value, torch.tensor([-1., 0., 0., 1.]), atol=0, rtol=0)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_zero_groups_exact(bits, dtype):
    x = torch.zeros(2, 3, 7, dtype=dtype)
    result = fake_quantize(x, bits, 4)
    assert torch.equal(result.value, x)
    assert result.value.dtype == dtype
    assert result.details["scales"].shape == (2, 3, 2)
    assert torch.equal(result.details["scales"], torch.ones(2, 3, 2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_16_is_native_identity_with_actual_element_size(dtype):
    base = torch.arange(24, dtype=dtype).reshape(3, 8)
    x = base[:, ::2]
    result = fake_quantize(x, 16, 3)
    assert result.value is x
    assert result.payload_bytes == x.numel() * x.element_size()
    assert result.logical_bytes == result.payload_bytes
    assert result.scale_bytes == result.metadata_bytes == 0
    assert result.physical_bytes == base.untyped_storage().nbytes()
    assert result.details["precision_kind"] == "native_identity"
    assert result.details["accounting_kind"] == "native_input_element_size"


def test_bf16_determinism_no_mutation_no_rng_or_graph_changes():
    x = torch.tensor([[[.125, -.375, .75, 1., 1.5], [2., -.25, .125, 3., 4.]]], dtype=torch.bfloat16).transpose(0, 1)
    x.requires_grad_(True)
    before = x.detach().clone()
    rng = torch.get_rng_state().clone()
    first = fake_quantize(x, 3, 3)
    second = fake_quantize(x, 3, 3)
    assert torch.equal(before, x)
    assert torch.equal(rng, torch.get_rng_state())
    assert x.grad is None and first.value.grad_fn is None
    assert torch.equal(first.value, second.value)
    assert torch.equal(first.details["scales"], second.details["scales"])
    expected = torch.empty_like(x, dtype=torch.float32)
    for start in range(0, x.shape[-1], 3):
        block = x.float()[..., start:start + 3]
        scale = block.abs().amax(dim=-1, keepdim=True) / 3
        expected[..., start:start + 3] = (block / scale).round().clamp(-3, 3) * scale
    torch.testing.assert_close(first.value.float(), expected.to(torch.bfloat16).float(), rtol=0, atol=0)
    assert torch.linalg.vector_norm(first.value.float() - x.float()) > 0


def test_input_is_quantized_directly_without_a_predictor_residual():
    x = torch.tensor([10., 11., 12., 13.])
    direct = fake_quantize(x, 2)
    residual_example = 10 + fake_quantize(x - 10, 2).value
    assert torch.equal(direct.value, torch.full_like(x, 13))
    assert not torch.equal(direct.value, residual_example)
    assert "no predictor residual" in direct.details["error_source"]


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_largest_finite_fp32_stays_finite(bits):
    limit = torch.finfo(torch.float32).max
    x = torch.tensor([-limit, limit])
    assert torch.isfinite(fake_quantize(x, bits).value).all()


def test_empty_leading_dimensions_and_tiny_fp64():
    result = fake_quantize(torch.empty(0, 3, 7), 3, 4)
    assert result.value.shape == (0, 3, 7)
    assert result.payload_bytes == result.scale_bytes == result.physical_bytes == 0
    assert result.metadata_bytes == 56
    tiny = fake_quantize(torch.tensor([1e-300, -1e-300], dtype=torch.float64), 4)
    assert torch.equal(tiny.value, torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="FP32 quantization scales"):
        fake_quantize(torch.tensor([1e300], dtype=torch.float64), 4)


@pytest.mark.parametrize("x", [torch.tensor(1.), torch.empty(2, 0), torch.tensor([float("nan")]), torch.tensor([float("inf")])])
def test_invalid_tensor_values_or_shape(x):
    with pytest.raises(ValueError):
        fake_quantize(x, 4)


@pytest.mark.parametrize("x", [None, [1.], torch.tensor([1]), torch.tensor([1 + 2j])])
def test_nonfloating_inputs_rejected(x):
    with pytest.raises(TypeError):
        fake_quantize(x, 4)


@pytest.mark.parametrize("bits", [True, 1, 9, 15, 17, 4., "4"])
def test_invalid_precision_labels(bits):
    with pytest.raises(ValueError):
        fake_quantize(torch.ones(2), bits)


@pytest.mark.parametrize("size", [True, 0, -1, 2., None])
def test_invalid_group_sizes(size):
    with pytest.raises(ValueError):
        fake_quantize(torch.ones(2), 4, size)


def options():
    return {
        "k": [PrecisionOption("k4", 4, 5, {"bits": 4, "role": "k"}), PrecisionOption("k8", 8, 1, {"bits": 8, "role": "k"})],
        "v": [PrecisionOption("v4", 4, 4, {"bits": 4, "role": "v"}), PrecisionOption("v8", 8, 2, {"bits": 8, "role": "v"})],
    }


def names(result):
    return tuple(option.name for _, option in sorted(result["assignment"].items()))


def test_small_exhaustive_additive_optimum_matches_bruteforce():
    choices = options()
    result = allocate(choices, 12)
    feasible = [(sum(o.error for o in combination), sum(o.bytes for o in combination), tuple(o.name for o in combination))
                for combination in itertools.product(*choices.values()) if sum(o.bytes for o in combination) <= 12]
    assert (result["error"], result["total_bytes"], names(result)) == min(feasible)
    assert names(result) == ("k8", "v4")
    assert result["error"] == 5 and result["total_bytes"] == 12
    assert result["objective_kind"] == "additive_diagnostic_proxy"
    assert result["states_evaluated"] == 3 and result["total_states"] == 4
    assert result["optimality"] == "exhaustive_feasible_assignments"


def test_joint_nonadditive_callback_can_choose_a_different_assignment():
    measured_joint_losses = {("k4", "v4"): 8., ("k8", "v4"): 9., ("k4", "v8"): .5}
    visited = []
    def objective(assignment):
        key = tuple(assignment[group].name for group in ("k", "v"))
        visited.append(key)
        # A callback may mutate its dictionary without corrupting results.
        assignment.clear()
        return measured_joint_losses[key]
    result = allocate(options(), 12, objective=objective)
    assert names(result) == ("k4", "v8")
    assert result["error"] == .5 and result["objective_kind"] == "callback_evaluated"
    assert len(visited) == result["states_evaluated"] == 3


def test_frontier_uses_actual_option_cost_and_removes_dominated_assignments():
    result = allocation_frontier(options())
    assert [(row["total_bytes"], row["error"], names(row)) for row in result["rows"]] == [
        (8, 9, ("k4", "v4")), (12, 5, ("k8", "v4")), (16, 3, ("k8", "v8")),
    ]
    assert result["states_evaluated"] == result["total_states"] == 4
    assert result["objective_kind"] == "additive_diagnostic_proxy"
    budgeted = allocation_frontier(options(), 12)
    assert len(budgeted["rows"]) == 2 and budgeted["states_evaluated"] == 3
    quantized = fake_quantize(torch.ones(1, 65), 3, 64)
    costed = allocate({"g": [PrecisionOption("3-bit", quantized.logical_bytes, .25)]}, quantized.logical_bytes)
    assert costed["total_bytes"] == quantized.payload_bytes + quantized.scale_bytes + quantized.metadata_bytes


def test_frontier_callback_and_equal_points_keep_all_assignments():
    values = {"g": [PrecisionOption("a", 2, 1), PrecisionOption("b", 2, 1), PrecisionOption("c", 3, 0)]}
    result = allocation_frontier(values, objective=lambda _: .5)
    assert [names(row) for row in result["rows"]] == [("a",), ("b",)]
    assert result["objective_kind"] == "callback_evaluated"
    assert names(allocate(values, 3, objective=lambda _: .5)) == ("a",)


def test_policy_rows_include_uniform_and_k_higher_at_actual_costs():
    result = precision_policy_rows(options())
    assert [(row["policy"], names(row), row["total_bytes"]) for row in result["rows"]] == [
        ("uniform", ("k4", "v4"), 8), ("k_higher", ("k8", "v4"), 12), ("uniform", ("k8", "v8"), 16),
    ]
    assert all(row["objective_kind"] == "additive_diagnostic_proxy" for row in result["rows"])
    assert result["states_evaluated"] == 4
    with pytest.raises(ValueError, match="metadata"):
        precision_policy_rows({"g": [PrecisionOption("unlabelled", 2, 1)]})


def test_empty_group_mapping_has_one_empty_assignment():
    result = allocate({}, 0)
    assert result["assignment"] == {} and result["total_bytes"] == result["error"] == 0
    assert result["states_evaluated"] == result["total_states"] == 1
    frontier = allocation_frontier({})
    assert len(frontier["rows"]) == 1 and frontier["rows"][0]["assignment"] == {}
    assert allocate({}, 0, objective=lambda assignment: len(assignment))["error"] == 0


@pytest.mark.parametrize("solver", [allocate, allocation_frontier])
def test_infeasible_budget_and_empty_option_list_are_explicit(solver):
    with pytest.raises(ValueError, match="no feasible assignment"):
        solver(options(), 7)
    with pytest.raises(ValueError, match="nonempty sequence"):
        solver({"k": []}, 12)


@pytest.mark.parametrize("solver", [allocate, allocation_frontier, precision_policy_rows])
def test_search_limit_is_not_a_greedy_optimum(solver):
    calls = []
    with pytest.raises(ValueError, match="state space.*no approximate optimum"):
        solver(options(), 12, max_states=3, objective=lambda assignment: calls.append(assignment) or 0)
    assert calls == []


@pytest.mark.parametrize("kwargs", [
    {"budget_bytes": -1}, {"budget_bytes": 1.5}, {"budget_bytes": True},
    {"budget_bytes": 10, "max_states": 0}, {"budget_bytes": 10, "max_states": True},
    {"budget_bytes": 10, "max_states": 3.5},
])
def test_invalid_solver_budget_or_state_limit(kwargs):
    with pytest.raises(ValueError):
        allocate(options(), **kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0", True, None])
def test_invalid_callback_output(bad):
    with pytest.raises(ValueError, match="finite real scalar"):
        allocate(options(), 8, objective=lambda _: bad)


def test_invalid_solver_structure_and_callback():
    with pytest.raises(TypeError, match="mapping"):
        allocate([], 0)
    with pytest.raises(ValueError, match="group names"):
        allocate({1: [PrecisionOption("a", 0, 0)]}, 0)
    with pytest.raises(TypeError, match="non-PrecisionOption"):
        allocate({"g": [1]}, 0)
    with pytest.raises(ValueError, match="duplicate"):
        allocate({"g": [PrecisionOption("a", 0, 0), PrecisionOption("a", 1, 1)]}, 1)
    with pytest.raises(TypeError, match="callable"):
        allocate(options(), 10, objective="sum")
    with pytest.raises(ValueError, match="requires"):
        allocate(options(), None)
    with pytest.raises(TypeError, match="mapping"):
        precision_policy_rows([], 0)


@pytest.mark.parametrize("args", [
    ("", 1, 0), ("x", -1, 0), ("x", True, 0), ("x", 1.5, 0),
    ("x", 1, -1), ("x", 1, float("nan")), ("x", 1, float("inf")), ("x", 1, True),
])
def test_invalid_precision_options(args):
    with pytest.raises(ValueError):
        PrecisionOption(*args)


def test_option_metadata_is_copied_and_optional():
    metadata = {"bits": 4}
    option = PrecisionOption("x", 1, 0, metadata)
    metadata["bits"] = 8
    assert option.metadata == {"bits": 4}
    assert PrecisionOption("x", 1, 0, None).metadata == {}
    with pytest.raises(TypeError, match="metadata"):
        PrecisionOption("x", 1, 0, [])
