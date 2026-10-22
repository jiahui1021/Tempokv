"""Independent fake quantization and exhaustive precision-budget diagnostics.

The quantizer operates on the supplied tensor itself.  It has no predictor,
history, token-selection policy, random sampling, or production-cache codec.
Callers must select eligible historical tokens before calling it.

Low-bit accounting describes a hypothetical packed format: the payload is
``ceil(numel * bits / 8)`` bytes, scales are FP32, and metadata comprises a
16-byte header, 8 bytes per shape dimension, and a 16-byte group descriptor.
There is no row alignment or per-group padding in that logical payload.  This
module does not allocate that packed format.  ``physical_bytes`` counts unique
backing storages of the returned floating value and exposed scale tensor;
Python objects and transient working tensors are outside that count.  Thus it
must not be reported as a realized packed-cache memory saving.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor


METADATA_HEADER_BYTES = 16
METADATA_SHAPE_DIM_BYTES = 8
METADATA_GROUP_DESCRIPTOR_BYTES = 16


@dataclass(frozen=True)
class FakeQuantized:
    value: Tensor
    payload_bytes: int
    scale_bytes: int
    metadata_bytes: int
    logical_bytes: int
    physical_bytes: int
    details: dict[str, Any]


def _storage_bytes(*tensors: Tensor) -> int:
    """Count backing allocations once, including a view's complete storage."""
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        identity = (str(tensor.device), storage.data_ptr(), size)
        if identity not in seen:
            seen.add(identity)
            total += size
    return total


def fake_quantize(x: Tensor, bits: int, group_size: int = 64) -> FakeQuantized:
    """Return symmetric per-last-axis-group quantization in ``x``'s dtype.

    Integer codes use the symmetric range ``[-qmax, qmax]`` with
    ``qmax = 2**(bits - 1) - 1`` (one signed code is unused).  FP32 scales are
    ``max(abs(group)) / qmax``; zero groups use scale one.  Nonzero scales are
    bounded below by the smallest normal FP32 value, so tiny FP64 groups can
    round to zero.  Values too large for a finite FP32 scale are rejected.
    ``torch.round`` supplies deterministic round-to-nearest, ties-to-even.

    ``bits=16`` is an identity control, even for FP32/FP64 input: it returns the
    original tensor and counts its actual native element size, with no scale
    or quantization metadata.  The label does not imply conversion to FP16.
    Empty leading dimensions are supported; the feature dimension must be
    positive.  The input, its RNG state, and its gradient graph are untouched.
    """
    if not isinstance(x, Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating torch.Tensor")
    if x.layout != torch.strided:
        raise ValueError("x must be a strided tensor")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("x needs a positive last feature dimension")
    if type(bits) is not int or bits not in (*range(2, 9), 16):
        raise ValueError("bits must be an integer in 2..8, or 16 for native identity")
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain only finite values")

    native_bytes = x.numel() * x.element_size()
    common = {
        "bits": bits,
        "group_size": group_size,
        "shape": tuple(x.shape),
        "native_dtype": str(x.dtype),
        "native_element_bytes": x.element_size(),
        "input_value_bytes": native_bytes,
        "physical_bytes_basis": "unique returned tensor backing storages; excludes Python and transient allocations",
        "packed_storage_allocated": False,
        "error_source": "direct tensor reconstruction; no predictor residual",
    }
    if bits == 16:
        details = {
            **common,
            "precision_kind": "native_identity",
            "accounting_kind": "native_input_element_size",
            "scale_count": 0,
            "physical_value_bytes": x.untyped_storage().nbytes(),
            "physical_scale_bytes": 0,
        }
        return FakeQuantized(x, native_bytes, 0, 0, native_bytes, _storage_bytes(x), details)

    width = x.shape[-1]
    rows_count = math.prod(x.shape[:-1])
    groups_per_row = (width + group_size - 1) // group_size
    qmax = 2 ** (bits - 1) - 1
    working_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    # Detaching keeps this research intervention from retaining training graphs.
    rows = x.detach().to(working_dtype).reshape(rows_count, width)
    restored = torch.empty_like(rows)
    scale_rows = torch.empty((rows_count, groups_per_row), dtype=torch.float32, device=x.device)
    minimum_scale = torch.finfo(torch.float32).tiny
    output_limit = torch.finfo(x.dtype).max
    for group in range(groups_per_row):
        start, stop = group * group_size, min((group + 1) * group_size, width)
        block = rows[:, start:stop]
        maximum = block.abs().amax(dim=-1)
        scales = torch.where(maximum == 0, torch.ones_like(maximum), (maximum / qmax).clamp_min(minimum_scale)).to(torch.float32)
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("x is too large for finite FP32 quantization scales")
        scale_rows[:, group] = scales
        work_scales = scales.to(working_dtype).unsqueeze(-1)
        codes = torch.round(block / work_scales).clamp(-qmax, qmax)
        # Near a dtype's largest value, scale rounding can overflow the product.
        restored[:, start:stop] = (codes * work_scales).clamp(-output_limit, output_limit)

    value = restored.reshape(x.shape).to(x.dtype)
    scales = scale_rows.reshape((*x.shape[:-1], groups_per_row))
    payload_bytes = (x.numel() * bits + 7) // 8
    scale_bytes = scale_rows.numel() * 4
    metadata_layout = {
        "header_bytes": METADATA_HEADER_BYTES,
        "shape_bytes": METADATA_SHAPE_DIM_BYTES * x.ndim,
        "group_descriptor_bytes": METADATA_GROUP_DESCRIPTOR_BYTES,
    }
    metadata_bytes = sum(metadata_layout.values())
    details = {
        **common,
        "precision_kind": "symmetric_group_fake_quantization",
        "accounting_kind": "hypothetical_contiguous_bit_packing",
        "rounding": "nearest_ties_to_even",
        "qmin": -qmax,
        "qmax": qmax,
        "row_count": rows_count,
        "groups_per_row": groups_per_row,
        "scale_count": scales.numel(),
        "scales": scales,
        "scale_dtype": "torch.float32",
        "metadata_layout": metadata_layout,
        "physical_value_bytes": value.untyped_storage().nbytes(),
        "physical_scale_bytes": scales.untyped_storage().nbytes(),
    }
    return FakeQuantized(value, payload_bytes, scale_bytes, metadata_bytes,
                         payload_bytes + scale_bytes + metadata_bytes,
                         _storage_bytes(value, scales), details)


@dataclass(frozen=True)
class PrecisionOption:
    """One explicit byte cost and measured error (or diagnostic proxy).

    ``bytes`` must include every cost the caller wants constrained, including
    scales and metadata.  The solver does not infer costs from a bit label.
    Optional ``metadata`` can include ``bits`` and ``role`` (``k`` or ``v``) for
    policy comparisons.  Neither field changes the optimization objective.
    """
    name: str
    bytes: int
    error: float
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("option name must be a nonempty string")
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("option bytes must be a nonnegative integer")
        if isinstance(self.error, bool) or not isinstance(self.error, Real) or not math.isfinite(self.error) or self.error < 0:
            raise ValueError("option error must be finite and nonnegative")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("option metadata must be a dict or None")
        object.__setattr__(self, "error", float(self.error))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


Assignment = dict[str, PrecisionOption]
Objective = Callable[[Assignment], float]
OBJECTIVE_PROXY = "additive_diagnostic_proxy"
OBJECTIVE_CALLBACK = "callback_evaluated"


def _validate_options(options_by_group: Mapping[str, Sequence[PrecisionOption]],
                      budget_bytes: int | None, max_states: int) -> tuple[list[str], list[list[PrecisionOption]], int]:
    if not isinstance(options_by_group, Mapping):
        raise TypeError("options_by_group must be a mapping")
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes < 0):
        raise ValueError("budget_bytes must be a nonnegative integer")
    if type(max_states) is not int or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if any(not isinstance(key, str) or not key for key in options_by_group):
        raise ValueError("group names must be nonempty strings")
    groups = sorted(options_by_group)
    choices = []
    total_states = 1
    for group in groups:
        values = options_by_group[group]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"group {group!r} needs a nonempty sequence of precision options")
        if any(not isinstance(option, PrecisionOption) for option in values):
            raise TypeError(f"group {group!r} contains a non-PrecisionOption value")
        if len({option.name for option in values}) != len(values):
            raise ValueError(f"group {group!r} has duplicate option names")
        choices.append(sorted(values, key=lambda option: (option.bytes, option.error, option.name)))
        total_states *= len(values)
    if total_states > max_states:
        raise ValueError(f"state space ({total_states}) exceeds max_states={max_states}; no approximate optimum returned")
    return groups, choices, total_states


def _assignment_key(assignment: Assignment) -> tuple[str, ...]:
    return tuple(assignment[group].name for group in sorted(assignment))


def _enumerate(options_by_group: Mapping[str, Sequence[PrecisionOption]], budget_bytes: int | None,
               max_states: int, objective: Objective | None) -> tuple[list[dict[str, Any]], int, str]:
    groups, choices, total_states = _validate_options(options_by_group, budget_bytes, max_states)
    if objective is not None and not callable(objective):
        raise TypeError("objective must be callable or None")
    objective_kind = OBJECTIVE_PROXY if objective is None else OBJECTIVE_CALLBACK
    rows = []
    for combination in product(*choices):
        total_bytes = sum(option.bytes for option in combination)
        if budget_bytes is not None and total_bytes > budget_bytes:
            continue
        assignment = dict(zip(groups, combination))
        error = math.fsum(option.error for option in combination) if objective is None else objective(dict(assignment))
        if isinstance(error, bool) or not isinstance(error, Real) or not math.isfinite(error):
            raise ValueError("objective must return a finite real scalar")
        rows.append({"assignment": assignment, "total_bytes": total_bytes,
                     "error": float(error), "objective_kind": objective_kind})
    if not rows:
        raise ValueError(f"no feasible assignment within budget_bytes={budget_bytes}")
    return rows, total_states, objective_kind


def allocate(options_by_group: Mapping[str, Sequence[PrecisionOption]], budget_bytes: int,
             max_states: int = 4096, objective: Objective | None = None) -> dict[str, Any]:
    """Exhaustively minimize error within the supplied actual byte budget.

    Without ``objective`` the sum of independent option errors is explicitly an
    additive diagnostic proxy, not a joint attention-loss optimum.  A callback
    receives the complete assignment and can evaluate its non-additive loss.
    Equal objectives prefer fewer bytes, then lexicographic option names.

    Every Cartesian-product state counts toward ``max_states``, even if its
    byte cost exceeds the budget.  Oversized searches and infeasible budgets
    raise ``ValueError``; no greedy result is presented as optimal.  An empty
    group mapping has the single empty, zero-byte assignment.
    """
    if budget_bytes is None:
        raise ValueError("allocate requires a nonnegative integer budget_bytes")
    rows, total_states, objective_kind = _enumerate(options_by_group, budget_bytes, max_states, objective)
    best = min(rows, key=lambda row: (row["error"], row["total_bytes"], _assignment_key(row["assignment"])))
    return {**best, "states_evaluated": len(rows), "total_states": total_states,
            "objective_kind": objective_kind, "optimality": "exhaustive_feasible_assignments"}


def allocation_frontier(options_by_group: Mapping[str, Sequence[PrecisionOption]],
                        budget_bytes: int | None = None, max_states: int = 4096,
                        objective: Objective | None = None) -> dict[str, Any]:
    """Return all nondominated byte/error assignments under an optional budget.

    Each row uses the exact sum of ``PrecisionOption.bytes``.  A point is
    dominated only when another assignment is no worse in both cost and error
    and strictly better in at least one.  Equal points retain every assignment.
    Validation, proxy labels, infeasible behavior and search limits match
    :func:`allocate`; ``budget_bytes=None`` considers all combinations.
    """
    rows, total_states, objective_kind = _enumerate(options_by_group, budget_bytes, max_states, objective)
    rows.sort(key=lambda row: (row["total_bytes"], row["error"], _assignment_key(row["assignment"])))
    frontier = []
    best_error = math.inf
    index = 0
    while index < len(rows):
        end = index + 1
        while end < len(rows) and rows[end]["total_bytes"] == rows[index]["total_bytes"]:
            end += 1
        cheapest_error = rows[index]["error"]
        if cheapest_error < best_error:
            frontier.extend(row for row in rows[index:end] if row["error"] == cheapest_error)
            best_error = cheapest_error
        index = end
    return {"rows": frontier, "states_evaluated": len(rows), "total_states": total_states,
            "objective_kind": objective_kind, "optimality": "exhaustive_feasible_assignments"}


def precision_policy_rows(options_by_group: Mapping[str, Sequence[PrecisionOption]],
                          budget_bytes: int | None = None, max_states: int = 4096,
                          objective: Objective | None = None) -> dict[str, Any]:
    """Compare all feasible uniform and K-higher assignments, including dominated ones.

    Each option must label ``metadata['bits']`` (2..8 or 16) and
    ``metadata['role']`` (``k`` or ``v``).  Uniform means every group has the same
    label.  K-higher means every K label exceeds every V label.  The latter
    requires at least one group of each role.  Native identity remains labelled
    16; its actual byte cost is still provided independently by the caller.
    Default scores retain the additive diagnostic proxy label.
    """
    _validate_options(options_by_group, budget_bytes, max_states)
    for options in options_by_group.values():
        for option in options:
            if not isinstance(option, PrecisionOption):
                continue  # The shared validator supplies the consistent error.
            bits = option.metadata.get("bits")
            role = option.metadata.get("role")
            if type(bits) is not int or bits not in (*range(2, 9), 16) or role not in {"k", "v"}:
                raise ValueError("policy comparisons require option metadata bits=2..8/16 and role=k/v")
    rows, total_states, objective_kind = _enumerate(options_by_group, budget_bytes, max_states, objective)
    selected = []
    for row in rows:
        options = list(row["assignment"].values())
        bits = {option.metadata["bits"] for option in options}
        k_bits = [option.metadata["bits"] for option in options if option.metadata["role"] == "k"]
        v_bits = [option.metadata["bits"] for option in options if option.metadata["role"] == "v"]
        policy = "uniform" if len(bits) == 1 else "k_higher" if k_bits and v_bits and min(k_bits) > max(v_bits) else None
        if policy is not None:
            selected.append({**row, "policy": policy,
                             "k_bits": sorted(set(k_bits)), "v_bits": sorted(set(v_bits))})
    selected.sort(key=lambda row: (row["total_bytes"], row["error"], _assignment_key(row["assignment"])))
    return {"rows": selected, "states_evaluated": len(rows), "total_states": total_states,
            "objective_kind": objective_kind}
