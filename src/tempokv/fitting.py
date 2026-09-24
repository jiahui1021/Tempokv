"""Fit the small linear predictors consumed by :mod:`tempokv.codec`.

The routines in this module deliberately operate on reconstructed history.
That keeps fitting consistent with decoding, where clean past frames are not
available.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import torch

from .attention_objective import (
    cross_kv_rollout,
    normalized_errors,
    objective_terms,
    phase_score,
    temporal_rollout,
)
from .codec import _cuda_matmul_precision, _dtype, anchor_roundtrip as _anchor_roundtrip
from .config import (
    FIXED_CROSS_KV_V_PREDICTOR_MODES,
    CodecConfig,
    fixed_temporal_predictor_mode,
    is_cross_kv_v_predictor,
    is_fixed_temporal_predictor,
    is_fixed_v_predictor,
    is_temporal_v_predictor,
)
from .parameters import CrossKVParameters, TemporalParameters
from .quantization import roundtrip


_FIXED_TEMPORAL_DEGREES = {
    "identity": 0,
    "taylor": 1,
    "taylor2": 2,
}


@torch.no_grad()
def build_fixed_temporal_parameters(
    predictor_mode: str,
    *,
    history_length: int,
    dim: int,
    stride: int | None = None,
    device: torch.device | str = "cpu",
    signature: str | None = None,
    layer_ids: tuple[int, ...] | None = None,
    head_count: int | None = None,
) -> TemporalParameters:
    """Build deterministic temporal coefficients for predictor ablations.

    The history is ordered newest-to-oldest.  ``identity`` is a zero-order
    hold, while ``taylor`` and ``taylor2`` use one- and two-step backward finite
    differences respectively:

    ``x[t+1] = x[t]``;
    ``x[t+1] = 2*x[t] - x[t-1]``; and
    ``x[t+1] = 3*x[t] - 3*x[t-1] + x[t-2]``.

    Warm-up rows use the highest order supported by the history available at
    that point.  The returned object has the same layout as fitted temporal
    parameters, so the encoder, decoder, Triton path, and parameter checks all
    exercise the identical recurrence.
    """
    degree = _FIXED_TEMPORAL_DEGREES[predictor_mode]
    if history_length < degree + 1:
        raise ValueError(
            f"predictor_mode={predictor_mode} needs history_length >= {degree + 1}"
        )
    base_weights = torch.zeros(
        history_length, history_length, dim, dtype=torch.float32, device=device
    )
    for order in range(1, history_length + 1):
        active_degree = min(degree, order - 1)
        coefficients = torch.tensor(
            [
                ((-1) ** lag) * math.comb(active_degree + 1, lag + 1)
                for lag in range(active_degree + 1)
            ],
            dtype=base_weights.dtype,
            device=base_weights.device,
        )
        base_weights[order - 1, : active_degree + 1] = coefficients[:, None]

    if layer_ids is None:
        weights = base_weights
        bias = torch.zeros(history_length, dim, dtype=torch.float32, device=device)
    else:
        weights = base_weights[:, :, None, :].expand(
            history_length, history_length, head_count, dim
        )
        weights = weights.unsqueeze(0).expand(
            len(layer_ids), history_length, history_length, head_count, dim
        ).clone()
        bias = torch.zeros(
            len(layer_ids), history_length, head_count, dim,
            dtype=torch.float32,
            device=device,
        )
    return TemporalParameters(
        history_weights=weights,
        history_bias=bias,
        stride=stride,
        layer_ids=layer_ids,
        codec_signature=signature,
    )


@torch.no_grad()
def _attach_fixed_cross_kv_history(
    cross: CrossKVParameters,
    predictor_mode: str,
    *,
    history_length: int,
) -> CrossKVParameters:
    """Attach fixed diagonal V-history coefficients to a fitted CrossKV map.

    ``CrossKVParameters`` stores one coefficient vector per available lag rather
    than warm-up rows.  The final fixed temporal row is therefore used here;
    missing warm-up lags are zero-padded by the runtime, matching the existing
    CrossKV recurrence contract.
    """
    fixed = build_fixed_temporal_parameters(
        fixed_temporal_predictor_mode(predictor_mode),
        history_length=history_length,
        dim=int(cross.weight.shape[-1]),
        device=cross.weight.device,
        signature=cross.codec_signature,
    )
    coefficients = fixed.history_weights[-1]
    if cross.layered:
        history_weights = coefficients[None, :, None, :].expand(
            int(cross.weight.shape[0]),
            history_length,
            int(cross.weight.shape[1]),
            int(cross.weight.shape[-1]),
        ).clone()
    else:
        history_weights = coefficients[:, None, :].expand(
            history_length,
            int(cross.weight.shape[0]),
            int(cross.weight.shape[-1]),
        ).clone()
    return CrossKVParameters(
        weight=cross.weight,
        bias=cross.bias,
        stride=cross.stride,
        layer_ids=cross.layer_ids,
        codec_signature=cross.codec_signature,
        parameter_set_id=cross.parameter_set_id,
        coordinate_system=cross.coordinate_system,
        history_weights=history_weights,
        projection_source=cross.projection_source,
    )


def _fixed_history_term(
    history_record: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a fixed V-history recurrence on one legacy fitting record."""
    frame_count = int(history_record.shape[1])
    history_length = int(coefficients.shape[0])
    term = torch.zeros_like(history_record, dtype=torch.float32)
    for frame in range(1, frame_count):
        for lag in range(1, min(frame, history_length) + 1):
            previous = history_record[:, frame - lag].float()
            coefficient = coefficients[lag - 1]
            while coefficient.ndim < previous.ndim:
                coefficient = coefficient.unsqueeze(0)
            term[:, frame] += previous * coefficient
    return term


def _subtract_fixed_cross_kv_history(
    clean_values: torch.Tensor,
    history_values: torch.Tensor,
    cross: CrossKVParameters,
) -> torch.Tensor:
    """Remove a fixed V-history term before fitting the CrossKV projection."""
    adjusted = clean_values.clone()
    for layer in range(int(clean_values.shape[0])):
        coefficients = cross.history_weights[layer]
        if clean_values.ndim == 7:
            for event in range(int(clean_values.shape[1])):
                value_record = clean_values[layer, event].movedim(-3, -4)
                history_record = history_values[layer, event].movedim(-3, -4)
                term = _fixed_history_term(history_record, coefficients)
                adjusted[layer, event].copy_(
                    (value_record.float() - term).movedim(-4, -3)
                )
        else:
            term = _fixed_history_term(history_values[layer], coefficients)
            adjusted[layer].copy_(clean_values[layer].float() - term)
    return adjusted


@torch.no_grad()
def _fit_cross_kv_fixed_history(
    reconstructed_keys: torch.Tensor,
    clean_values: torch.Tensor,
    *,
    config: CodecConfig,
    rounds: int,
    ridge: float,
    layer_ids: tuple[int, ...] | None,
    fit_history_source: str,
) -> CrossKVParameters:
    """Fit CrossKV while holding a fixed V-history recurrence constant."""

    def fit(values: torch.Tensor) -> CrossKVParameters:
        if config.fit_mode == "legacy":
            return fit_legacy_cross_kv(
                reconstructed_keys,
                values,
                ridge=ridge,
                stride=config.stride,
                layer_ids=layer_ids,
                weight_mode=config.cross_kv_weight_mode,
            )
        return fit_cross_kv(
            reconstructed_keys[..., 1:, :, :],
            values[..., 1:, :, :],
            ridge=ridge,
            stride=config.stride,
            layer_ids=layer_ids,
        )

    def attach(cross: CrossKVParameters) -> CrossKVParameters:
        return _attach_fixed_cross_kv_history(
            cross,
            config.v_predictor_mode,
            history_length=config.v_history_length,
        )

    cross = fit(clean_values)
    effective_rounds = rounds if fit_history_source == "reconstructed" else 1
    for _ in range(effective_rounds):
        fixed_cross = attach(cross)
        history_values = (
            _reconstruct_cross_kv_trace(
                reconstructed_keys, clean_values, fixed_cross, config
            )
            if fit_history_source == "reconstructed"
            else clean_values
        )
        cross = fit(
            _subtract_fixed_cross_kv_history(clean_values, history_values, fixed_cross)
        )
    return attach(cross)


def _compute_device_and_dtype(
    tensor: torch.Tensor,
) -> tuple[torch.device, torch.dtype]:
    """Use fast float32 equations on CUDA and stable float64 elsewhere."""
    if tensor.device.type == "cuda":
        return tensor.device, torch.float32
    if tensor.device.type == "mps":
        return torch.device("cpu"), torch.float64
    return tensor.device, torch.float64


def _pinv_rtol(dtype: torch.dtype) -> float:
    return 1e-6 if dtype == torch.float32 else 1e-10


def _legacy_layer_records(tensor: torch.Tensor, layer: int) -> list[torch.Tensor]:
    """Return legacy-protocol records as ``[B,F,P,H,D]`` in capture order.

    The legacy trace schema is ``[L,E,B,P,F,H,D]`` so ``F`` remains
    TempoKV's conventional ``-3`` temporal axis on disk.  The legacy fitter
    evaluates one event at a time in ``[B,F,P,H,D]``; restore that layout to
    preserve its reduction and FP64 accumulation order.  Smaller synthetic
    traces keep the established ``[L,...,F,H,D]`` layout and count as one
    record.
    """
    selected = tensor[layer]
    if tensor.ndim == 7:
        return [record.movedim(-3, -4).contiguous() for record in selected]
    if selected.ndim == 3:
        selected = selected.unsqueeze(0)
    return [selected]


def _solve(system: torch.Tensor, rhs: torch.Tensor, *, regularized: bool) -> torch.Tensor:
    """Solve batched normal equations, including singular unregularized fits."""
    if regularized:
        return torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.matmul(
        torch.linalg.pinv(system, rtol=_pinv_rtol(system.dtype)),
        rhs.unsqueeze(-1),
    ).squeeze(-1)



def _project_gain(
    weight: torch.Tensor, max_gain: float, *, lag_axis: int | None = None
) -> None:
    """Cap each channel's summed absolute history gain, in place."""
    if lag_axis is None:
        lag_axis = 2 if weight.ndim == 5 else 1
    gain = weight.abs().sum(dim=lag_axis, keepdim=True).clamp_min(1e-20)
    weight.mul_(torch.clamp(max_gain / gain, max=1.0))


def _objective_sums(
    keys: torch.Tensor,
    values: torch.Tensor,
    query_post_rope: torch.Tensor,
    key_post_rope: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    value_history_weights: torch.Tensor | None = None,
    *,
    config: CodecConfig,
    layer_chunk: int,
    backward_tensor: str | None = None,
    reference_sums: torch.Tensor | None = None,
    reconstruction_weight: float = 0.5,
    attention_weight: float = 0.5,
) -> torch.Tensor:
    """Accumulate the six objective sums over the trace, one layer chunk at a time.

    Chunking bounds the rollout activations kept for backward.  Native temporal
    and CrossKV parameters are sliced with the trace; shared parameters are
    reused.  Summing chunk gradients is the gradient of the whole trace.
    """
    totals = torch.zeros(6, dtype=torch.float64, device=keys.device)
    layer_count = int(keys.shape[0])
    for start in range(0, layer_count, layer_chunk):
        stop = min(start + layer_chunk, layer_count)
        chunk_key_weight = key_weight[start:stop] if key_weight.ndim == 5 else key_weight
        chunk_key_bias = key_bias[start:stop] if key_bias.ndim == 4 else key_bias
        rebuilt_key = temporal_rollout(
            keys[start:stop],
            chunk_key_weight,
            chunk_key_bias,
            bits=config.k_bits,
            config=config,
        )
        if is_temporal_v_predictor(config.v_predictor_mode):
            chunk_value_weight = (
                value_weight[start:stop] if value_weight.ndim == 5 else value_weight
            )
            chunk_value_bias = (
                value_bias[start:stop] if value_bias.ndim == 4 else value_bias
            )
            rebuilt_value = temporal_rollout(
                values[start:stop],
                chunk_value_weight,
                chunk_value_bias,
                bits=config.v_bits,
                config=config,
            )
        else:
            rebuilt_value = cross_kv_rollout(
                rebuilt_key,
                values[start:stop],
                value_weight[start:stop] if value_weight.ndim == 4 else value_weight,
                value_bias[start:stop] if value_bias.ndim == 3 else value_bias,
                (
                    value_history_weights[start:stop]
                    if value_weight.ndim == 4
                    else value_history_weights
                ),
                bits=config.v_bits,
                config=config,
            )
        terms = objective_terms(
            clean_key=keys[start:stop],
            clean_value=values[start:stop],
            rebuilt_key=rebuilt_key,
            rebuilt_value=rebuilt_value,
            query_post_rope=query_post_rope[start:stop],
            key_post_rope=key_post_rope[start:stop],
        )
        if backward_tensor is not None:
            if backward_tensor == "key":
                own_error = terms.key_error
                own_reference = reference_sums[1]
            else:
                own_error = terms.value_error
                own_reference = reference_sums[3]
            loss = reconstruction_weight * own_error / own_reference.clamp_min(
                1e-12
            ) + attention_weight * (
                terms.attention_error
                / reference_sums[5].clamp_min(1e-12)
            )
            if not torch.isfinite(loss):
                raise ValueError(
                    f"{backward_tensor} refinement produced a non-finite loss"
                )
            # The denominators cover the full trace, so summing chunk
            # gradients is exactly the gradient of the global objective.
            loss.backward(retain_graph=False)
        totals = totals + torch.stack(
            [term.detach().double() for term in terms.as_tuple()]
        )
    return totals


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """What one refinement phase did and why."""

    tensor: str
    accepted: bool
    best_step: int | None
    step_size: float | None
    steps_run: int
    start_score: float
    best_score: float
    errors: dict[str, float]


@torch.no_grad()
def _evaluate(
    keys: torch.Tensor,
    values: torch.Tensor,
    query_post_rope: torch.Tensor,
    key_post_rope: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    config: CodecConfig,
    layer_chunk: int,
) -> dict[str, float]:
    sums = _objective_sums(
        keys,
        values,
        query_post_rope,
        key_post_rope,
        parameters["key_weight"],
        parameters["key_bias"],
        parameters["value_weight"],
        parameters["value_bias"],
        parameters.get("value_history_weights"),
        config=config,
        layer_chunk=layer_chunk,
    )
    return normalized_errors(sums)


def refine_attention(
    keys: torch.Tensor,
    values: torch.Tensor,
    query_post_rope: torch.Tensor,
    key_post_rope: torch.Tensor,
    key_parameters: TemporalParameters,
    value_parameters: TemporalParameters | CrossKVParameters,
    config: CodecConfig,
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 10,
    min_relative_improvement: float = 1e-4,
    max_gain: float = 1.0,
    layer_chunk: int = 4,
    optimizer_name: str = "adam",
    reconstruction_weight: float = 0.5,
    attention_weight: float = 0.5,
    refine_tensors: Sequence[str] = ("key", "value"),
    sign_step_sizes: Sequence[float] = (0.01,),
) -> tuple[
    TemporalParameters,
    TemporalParameters | CrossKVParameters,
    dict[str, object],
]:
    """Refine least-squares predictors under an explicit ablation protocol.

    K and V are optimized in selected, separate phases.  Adam is the primary
    optimizer; ``sign_gradient`` reproduces the historical one-direction scan
    inside the same rollout and parameter protocol.  Candidate selection and
    guardrails use the same normalized objective weights as training.
    """
    if config.v_predictor_mode == "innovation":
        raise ValueError("attention refinement does not support innovation V")
    if is_fixed_v_predictor(config.v_predictor_mode):
        raise ValueError(
            f"attention refinement does not support fixed {config.v_predictor_mode} V"
        )
    phases_to_run = tuple(refine_tensors)
    if (
        config.k_predictor_mode == "direct"
        or is_fixed_temporal_predictor(config.k_predictor_mode)
    ) and "key" in phases_to_run:
        raise ValueError(
            f"K predictor {config.k_predictor_mode} has no trainable coefficients "
            "for attention refinement"
        )
    if (
        config.v_predictor_mode == "cross_kv"
        and config.cross_kv_weight_mode != "layer_head"
        and "value" in phases_to_run
    ):
        raise ValueError(
            "value refinement does not support "
            f"cross_kv_weight_mode={config.cross_kv_weight_mode} because it "
            "would update the "
            "broadcast CrossKV weights independently"
        )

    temporal_value = isinstance(value_parameters, TemporalParameters)
    current: dict[str, torch.Tensor] = {
        "key_weight": key_parameters.history_weights.detach().clone(),
        "key_bias": key_parameters.history_bias.detach().clone(),
    }
    if temporal_value:
        current.update(
            value_weight=value_parameters.history_weights.detach().clone(),
            value_bias=value_parameters.history_bias.detach().clone(),
        )
    else:
        current.update(
            value_weight=value_parameters.weight.detach().clone(),
            value_bias=value_parameters.bias.detach().clone(),
            value_history_weights=value_parameters.history_weights.detach().clone(),
        )
    baseline_sums = _objective_sums(
        keys,
        values,
        query_post_rope,
        key_post_rope,
        current["key_weight"],
        current["key_bias"],
        current["value_weight"],
        current["value_bias"],
        current.get("value_history_weights"),
        config=config,
        layer_chunk=layer_chunk,
    )
    baseline = normalized_errors(baseline_sums)

    def backward(tensor: str, trainable: dict[str, torch.Tensor]) -> None:
        _objective_sums(
            keys,
            values,
            query_post_rope,
            key_post_rope,
            **{**current, **trainable},
            config=config,
            layer_chunk=layer_chunk,
            backward_tensor=tensor,
            reference_sums=baseline_sums,
            reconstruction_weight=reconstruction_weight,
            attention_weight=attention_weight,
        )

    def project_gain(tensor: str, candidate: dict[str, torch.Tensor]) -> None:
        if tensor == "key":
            _project_gain(candidate["key_weight"], max_gain)
        elif temporal_value:
            _project_gain(candidate["value_weight"], max_gain)
        elif "value_history_weights" in candidate:
            _project_gain(candidate["value_history_weights"], max_gain, lag_axis=1)

    def score(
        tensor: str, candidate_values: dict[str, torch.Tensor]
    ) -> tuple[dict[str, float], float]:
        candidate_errors = _evaluate(
            keys,
            values,
            query_post_rope,
            key_post_rope,
            {**current, **candidate_values},
            config=config,
            layer_chunk=layer_chunk,
        )
        return candidate_errors, phase_score(
            candidate_errors,
            baseline,
            tensor=tensor,
            reconstruction_weight=reconstruction_weight,
            attention_weight=attention_weight,
        )

    phases: list[PhaseResult] = []
    errors = baseline
    for tensor in phases_to_run:
        parameter_names = [f"{tensor}_weight", f"{tensor}_bias"]
        if (
            tensor == "value"
            and "value_history_weights" in current
            and current["value_history_weights"].numel()
        ):
            parameter_names.append("value_history_weights")
        start_score = phase_score(
            errors,
            baseline,
            tensor=tensor,
            reconstruction_weight=reconstruction_weight,
            attention_weight=attention_weight,
        )
        start_errors = errors
        guarded_metrics = []
        if reconstruction_weight:
            guarded_metrics.append(f"{tensor}_nmse")
        if attention_weight:
            guarded_metrics.append("attention_nmse")

        def regressed(candidate_errors: dict[str, float]) -> bool:
            return any(
                candidate_errors[name] > start_errors[name] + 1e-12
                for name in guarded_metrics
            )

        best: tuple[
            float,
            int | None,
            float | None,
            dict[str, float],
            dict[str, torch.Tensor],
        ] | None = None
        trainable = {
            name: current[name].clone().requires_grad_(True)
            for name in parameter_names
        }

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(trainable.values(), lr=learning_rate)
            stale_steps = 0
            for step in range(1, steps + 1):
                steps_run = step
                optimizer.zero_grad(set_to_none=True)
                backward(tensor, trainable)
                optimizer.step()
                with torch.no_grad():
                    project_gain(tensor, trainable)
                candidate_values = {
                    name: parameter.detach().clone()
                    for name, parameter in trainable.items()
                }
                candidate_errors, candidate_score = score(tensor, candidate_values)
                best_score = start_score if best is None else best[0]
                relative_improvement = (best_score - candidate_score) / max(
                    abs(best_score), 1e-12
                )
                if not regressed(candidate_errors) and candidate_score < best_score:
                    best = (
                        candidate_score, step, None, candidate_errors, candidate_values
                    )
                    if relative_improvement >= min_relative_improvement:
                        stale_steps = 0
                    else:
                        stale_steps += 1
                else:
                    stale_steps += 1
                if stale_steps >= patience:
                    break
        else:
            steps_run = 1
            backward(tensor, trainable)
            for step_size in sign_step_sizes:
                candidate_values = {
                    name: current[name] - step_size * trainable[name].grad.sign()
                    for name in parameter_names
                }
                project_gain(tensor, candidate_values)
                candidate_errors, candidate_score = score(tensor, candidate_values)
                best_score = start_score if best is None else best[0]
                if not regressed(candidate_errors) and candidate_score < best_score:
                    best = (
                        candidate_score, None, step_size, candidate_errors, candidate_values
                    )

        if best is None:
            phases.append(
                PhaseResult(
                    tensor,
                    False,
                    None,
                    None,
                    steps_run,
                    start_score,
                    start_score,
                    errors,
                )
            )
            continue
        best_score, best_step, step_size, errors, candidate_values = best
        current.update(candidate_values)
        phases.append(
            PhaseResult(
                tensor,
                True,
                best_step,
                step_size,
                steps_run,
                start_score,
                best_score,
                errors,
            )
        )

    report = {
        "baseline": baseline,
        "final": errors,
        "optimizer": optimizer_name,
        "training_objective": "normalized_weighted_reconstruction_attention",
        "selection_objective": "normalized_weighted_reconstruction_attention",
        "objective_weights": {
            "reconstruction": float(reconstruction_weight),
            "attention": float(attention_weight),
        },
        "refine_tensors": list(phases_to_run),
        "steps": int(steps) if optimizer_name == "adam" else None,
        "learning_rate": float(learning_rate) if optimizer_name == "adam" else None,
        "patience": int(patience) if optimizer_name == "adam" else None,
        "min_relative_improvement": float(min_relative_improvement),
        "sign_step_sizes": (
            [float(step_size) for step_size in sign_step_sizes]
            if optimizer_name == "sign_gradient"
            else None
        ),
        "phases": [
            {
                "tensor": phase.tensor,
                "outcome": "accepted" if phase.accepted else "guardrail_fallback",
                "best_step": phase.best_step,
                "step_size": phase.step_size,
                "steps_run": phase.steps_run,
                "start_score": phase.start_score,
                "score": phase.best_score,
                **{f"final_{name}": value for name, value in phase.errors.items()},
            }
            for phase in phases
        ],
    }
    refined_key = replace(
        key_parameters,
        history_weights=current["key_weight"],
        history_bias=current["key_bias"],
    )
    refined_value = (
        replace(
            value_parameters,
            history_weights=current["value_weight"],
            history_bias=current["value_bias"],
        )
        if temporal_value
        else replace(
            value_parameters,
            weight=current["value_weight"],
            bias=current["value_bias"],
            history_weights=current["value_history_weights"],
        )
    )
    refined_key, refined_value = stamp_parameter_identity(
        refined_key,
        refined_value,
        signature=config.format_signature(),
        coordinate_system=(
            key_parameters.coordinate_system or value_parameters.coordinate_system
        ),
    )
    return refined_key, refined_value, report


@torch.no_grad()
def fit_temporal(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    history_length: int,
    shared: bool = True,
    ridge: float = 0.0,
    max_gain: float = 1.0,
    stride: int | None = None,
) -> TemporalParameters:
    """Fit a diagonal autoregressive predictor from decoder-visible history.

    ``reconstructed`` and ``target`` use ``[..., T, H, D]`` layout.  Leading
    dimensions are independent samples, ``T`` is frame order, ``H`` is the
    number of heads and ``D`` is the per-head dimension.  Shared fitting pools
    heads and returns ``[P,P,D]`` weights; per-head fitting returns
    ``[P,P,H,D]``.  Row ``q-1`` contains the order-``q`` warm-up parameters,
    with unused lag positions kept at zero.

    Ridge is scaled by the average centered feature energy and shrinks toward
    persistence (lag 1 has coefficient 1).  The L1 gain across lags is capped
    independently for every head/channel.
    """
    frame_count, head_count, dim = map(int, reconstructed.shape[-3:])
    work_device, compute_dtype = _compute_device_and_dtype(reconstructed)
    history_data = reconstructed.detach().to(device=work_device, dtype=compute_dtype)
    targets = target.detach().to(device=work_device, dtype=compute_dtype)

    if shared:
        weights = torch.zeros(
            history_length, history_length, dim, dtype=compute_dtype, device=work_device
        )
        bias = torch.zeros(history_length, dim, dtype=compute_dtype, device=work_device)
    else:
        weights = torch.zeros(
            history_length,
            history_length,
            head_count,
            dim,
            dtype=compute_dtype,
            device=work_device,
        )
        bias = torch.zeros(
            history_length, head_count, dim, dtype=compute_dtype, device=work_device
        )

    for order in range(1, history_length + 1):
        # Warm-up rows q<P are used exactly at frame q.  The Pth row is reused
        # for every later frame, matching runtime order selection.
        stop = order + 1 if order < history_length else frame_count
        x = torch.stack(
            [history_data[..., order - lag : stop - lag, :, :] for lag in range(1, order + 1)],
            dim=-2,
        )
        y = targets[..., order:stop, :, :]

        if shared:
            x_grouped = x.reshape(1, -1, order, dim)
            y_grouped = y.reshape(1, -1, dim)
        else:
            x_grouped = x.movedim(-3, 0).reshape(head_count, -1, order, dim)
            y_grouped = y.movedim(-2, 0).reshape(head_count, -1, dim)

        x_mean = x_grouped.mean(dim=1)
        y_mean = y_grouped.mean(dim=1)
        xc = x_grouped - x_mean[:, None, :, :]
        yc = y_grouped - y_mean[:, None, :]
        gram = torch.einsum("gnpd,gnqd->gdpq", xc, xc)
        rhs = torch.einsum("gnpd,gnd->gdp", xc, yc)

        scale = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / order
        penalty = ridge * scale.clamp_min(torch.finfo(compute_dtype).eps)
        system = gram + penalty[..., None, None] * torch.eye(
            order, dtype=compute_dtype, device=work_device
        )
        persistence = torch.zeros(order, dtype=compute_dtype, device=work_device)
        persistence[0] = 1.0
        rhs = rhs + penalty[..., None] * persistence
        coefficient = _solve(system, rhs, regularized=ridge > 0.0)

        l1_gain = coefficient.abs().sum(dim=-1)
        gain_scale = torch.clamp(max_gain / l1_gain.clamp_min(1e-30), max=1.0)
        coefficient = coefficient * gain_scale[..., None]
        intercept = y_mean - torch.einsum("gpd,gdp->gd", x_mean, coefficient)

        if shared:
            weights[order - 1, :order] = coefficient[0].transpose(0, 1)
            bias[order - 1] = intercept[0]
        else:
            weights[order - 1, :order] = coefficient.permute(2, 0, 1)
            bias[order - 1] = intercept

    return TemporalParameters(
        history_weights=weights.to(device=reconstructed.device, dtype=torch.float32).contiguous(),
        history_bias=bias.to(device=reconstructed.device, dtype=torch.float32).contiguous(),
        stride=stride,
    )


@torch.no_grad()
def fit_legacy_shared_temporal(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    history_length: int,
    ridge: float = 1.0,
    max_gain: float = 1.0,
    stride: int | None = None,
) -> TemporalParameters:
    """Fit the legacy single shared K predictor across every layer and head.

    It accumulates the normal-equation statistics in FP64 even on CUDA.  The
    warm-up row for order ``q < P`` uses only frame ``q``; the final row uses
    every remaining frame.  Ridge shrinks each channel toward persistence and
    the sum of absolute history gains is capped per channel.
    """
    frame_count = int(target.shape[-3])
    dim = int(target.shape[-1])
    work_device = (
        torch.device("cpu")
        if reconstructed.device.type == "mps"
        else reconstructed.device
    )

    stats: list[dict[str, torch.Tensor | int]] = []
    for order in range(1, history_length + 1):
        stats.append(
            {
                "count": 0,
                "sum_x": torch.zeros(
                    order, dim, dtype=torch.float64, device=work_device
                ),
                "sum_y": torch.zeros(dim, dtype=torch.float64, device=work_device),
                "sum_xx": torch.zeros(
                    order, order, dim, dtype=torch.float64, device=work_device
                ),
                "sum_xy": torch.zeros(
                    order, dim, dtype=torch.float64, device=work_device
                ),
            }
        )

    # Stream one layer at a time so production traces do not need a second
    # full FP64 copy on the fitting device.
    for layer in range(int(target.shape[0])):
        rebuilt_records = _legacy_layer_records(reconstructed, layer)
        target_records = _legacy_layer_records(target, layer)
        for rebuilt_record, target_record in zip(
            rebuilt_records, target_records, strict=True
        ):
            rebuilt_layer = rebuilt_record.detach().to(work_device)
            target_layer = target_record.detach().to(work_device)
            for order in range(1, history_length + 1):
                stop = order + 1 if order < history_length else frame_count
                history = torch.stack(
                    [
                        rebuilt_layer[:, order - lag : stop - lag]
                        for lag in range(1, order + 1)
                    ],
                    dim=-2,
                )
                x = history.reshape(-1, order, dim).double()
                y = target_layer[:, order:stop].reshape(-1, dim).double()
                current = stats[order - 1]
                current["count"] = int(current["count"]) + int(x.shape[0])
                torch.as_tensor(current["sum_x"]).add_(x.sum(dim=0))
                torch.as_tensor(current["sum_y"]).add_(y.sum(dim=0))
                torch.as_tensor(current["sum_xx"]).add_(
                    torch.einsum("nid,njd->ijd", x, x)
                )
                torch.as_tensor(current["sum_xy"]).add_(
                    torch.einsum("nid,nd->id", x, y)
                )

    weights = torch.zeros(
        history_length,
        history_length,
        dim,
        dtype=torch.float32,
        device=work_device,
    )
    bias = torch.zeros(
        history_length, dim, dtype=torch.float32, device=work_device
    )
    for order, current in enumerate(stats, start=1):
        count = int(current["count"])
        sum_x = torch.as_tensor(current["sum_x"])
        sum_y = torch.as_tensor(current["sum_y"])
        mean_x = sum_x / count
        mean_y = sum_y / count
        covariance = torch.as_tensor(current["sum_xx"]) - torch.einsum(
            "id,jd->ijd", sum_x, sum_x
        ) / count
        cross = torch.as_tensor(current["sum_xy"]) - sum_x * mean_y[None]
        gram = covariance.permute(2, 0, 1)
        rhs = cross.permute(1, 0)
        scale = (
            torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1) / order
        ).clamp_min(1e-20)
        penalty = ridge * scale
        system = gram + penalty[:, None, None] * torch.eye(
            order, dtype=torch.float64, device=work_device
        )
        persistence = torch.zeros(
            order, dtype=torch.float64, device=work_device
        )
        persistence[0] = 1.0
        coefficient = torch.linalg.solve(
            system, (rhs + penalty[:, None] * persistence).unsqueeze(-1)
        ).squeeze(-1)
        gain = coefficient.abs().sum(dim=-1).clamp_min(1e-20)
        coefficient.mul_(torch.clamp(max_gain / gain, max=1.0)[:, None])
        weights[order - 1, :order] = coefficient.transpose(0, 1).float()
        bias[order - 1] = (
            mean_y - (mean_x.transpose(0, 1) * coefficient).sum(dim=-1)
        ).float()

    return TemporalParameters(
        weights.to(target.device).contiguous(),
        bias.to(target.device).contiguous(),
        stride=stride,
    )


@torch.no_grad()
def fit_legacy_layer_head_temporal(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    history_length: int,
    ridge: float = 1.0,
    max_gain: float = 1.0,
    stride: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> TemporalParameters:
    """Fit the legacy K predictor independently for every layer and head.

    The statistics, FP64 accumulation, warm-up rows, ridge toward persistence
    and per-channel gain cap are those of :func:`fit_legacy_shared_temporal`;
    only the normal equations are kept per ``(layer, head)`` instead of pooled.
    Returns ``[L,P,P,H,D]`` weights and ``[L,P,H,D]`` bias.
    """
    layer_count = int(target.shape[0])
    frame_count = int(target.shape[-3])
    heads = int(target.shape[-2])
    dim = int(target.shape[-1])
    work_device = (
        torch.device("cpu")
        if reconstructed.device.type == "mps"
        else reconstructed.device
    )

    weights = torch.zeros(
        layer_count,
        history_length,
        history_length,
        heads,
        dim,
        dtype=torch.float32,
        device=work_device,
    )
    bias = torch.zeros(
        layer_count, history_length, heads, dim, dtype=torch.float32, device=work_device
    )
    for layer in range(layer_count):
        stats: list[dict[str, torch.Tensor | int]] = [
            {
                "count": 0,
                "sum_x": torch.zeros(
                    heads, order, dim, dtype=torch.float64, device=work_device
                ),
                "sum_y": torch.zeros(
                    heads, dim, dtype=torch.float64, device=work_device
                ),
                "sum_xx": torch.zeros(
                    heads, order, order, dim, dtype=torch.float64, device=work_device
                ),
                "sum_xy": torch.zeros(
                    heads, order, dim, dtype=torch.float64, device=work_device
                ),
            }
            for order in range(1, history_length + 1)
        ]
        for rebuilt_record, target_record in zip(
            _legacy_layer_records(reconstructed, layer),
            _legacy_layer_records(target, layer),
            strict=True,
        ):
            rebuilt_layer = rebuilt_record.detach().to(work_device)
            target_layer = target_record.detach().to(work_device)
            for order in range(1, history_length + 1):
                stop = order + 1 if order < history_length else frame_count
                history = torch.stack(
                    [
                        rebuilt_layer[:, order - lag : stop - lag]
                        for lag in range(1, order + 1)
                    ],
                    dim=-2,
                )
                x = history.reshape(-1, heads, order, dim).double()
                y = target_layer[:, order:stop].reshape(-1, heads, dim).double()
                current = stats[order - 1]
                current["count"] = int(current["count"]) + int(x.shape[0])
                torch.as_tensor(current["sum_x"]).add_(x.sum(dim=0))
                torch.as_tensor(current["sum_y"]).add_(y.sum(dim=0))
                torch.as_tensor(current["sum_xx"]).add_(
                    torch.einsum("nhid,nhjd->hijd", x, x)
                )
                torch.as_tensor(current["sum_xy"]).add_(
                    torch.einsum("nhid,nhd->hid", x, y)
                )

        for order, current in enumerate(stats, start=1):
            count = int(current["count"])
            sum_x = torch.as_tensor(current["sum_x"])
            sum_y = torch.as_tensor(current["sum_y"])
            mean_x = sum_x / count
            mean_y = sum_y / count
            covariance = torch.as_tensor(current["sum_xx"]) - torch.einsum(
                "hid,hjd->hijd", sum_x, sum_x
            ) / count
            cross = torch.as_tensor(current["sum_xy"]) - sum_x * mean_y[:, None]
            gram = covariance.permute(0, 3, 1, 2)
            rhs = cross.permute(0, 2, 1)
            scale = (
                torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1) / order
            ).clamp_min(1e-20)
            penalty = ridge * scale
            system = gram + penalty[..., None, None] * torch.eye(
                order, dtype=torch.float64, device=work_device
            )
            persistence = torch.zeros(order, dtype=torch.float64, device=work_device)
            persistence[0] = 1.0
            coefficient = torch.linalg.solve(
                system, (rhs + penalty[..., None] * persistence).unsqueeze(-1)
            ).squeeze(-1)
            gain = coefficient.abs().sum(dim=-1).clamp_min(1e-20)
            coefficient.mul_(torch.clamp(max_gain / gain, max=1.0)[..., None])
            weights[layer, order - 1, :order] = coefficient.permute(2, 0, 1).float()
            bias[layer, order - 1] = (
                mean_y - (mean_x.permute(0, 2, 1) * coefficient).sum(dim=-1)
            ).float()

    return TemporalParameters(
        weights.to(target.device).contiguous(),
        bias.to(target.device).contiguous(),
        stride=stride,
        layer_ids=tuple(range(layer_count)) if layer_ids is None else layer_ids,
    )


@torch.no_grad()
def fit_cross_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    reconstructed_values: torch.Tensor | None = None,
    history_length: int = 0,
    history_mode: str = "dense",
    ridge: float = 0.0,
    stride: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> CrossKVParameters:
    """Fit current-K and reconstructed-V-history maps per layer/head.

    With ``history_length=0``, inputs use ``[L,...,H,D]`` and retain the legacy
    ``V = K @ W_K + b`` behavior.  With one or two history terms, inputs use
    ``[L,...,T,H,D]``; frame zero is an anchor and later frames are fitted as
    ``V_t = K_t @ W_K + sum(V_hat_(t-i) @ W_Vi) + b``.  Missing warm-up lags
    are zero.  ``reconstructed_values`` must supply decoder-visible history;
    callers that intentionally want teacher forcing can pass ``values``.
    ``history_mode="dense"`` fits a full ``D x D`` map per lag and preserves
    the legacy API.  ``history_mode="diagonal"`` jointly fits the dense K map
    with one history coefficient per lag and output channel.
    """
    # A zero-history fit is exactly the legacy K-only regression.  Keep that
    # fast path even when callers preselect diagonal mode for a later fit.
    resolved_history_mode = "dense" if history_length == 0 else history_mode
    history_trace: torch.Tensor | None = None
    feature_keys = keys
    targets = values
    if history_length:
        history_trace = reconstructed_values
        frame_count = int(keys.shape[-3])
        feature_keys = keys[..., 1:, :, :]
        targets = values[..., 1:, :, :]
    layer_count = int(keys.shape[0])
    head_count = int(keys.shape[-2])
    dim = int(keys.shape[-1])
    feature_dim = dim + history_length * (
        dim if resolved_history_mode == "dense" else 1
    )

    work_device, compute_dtype = _compute_device_and_dtype(keys)
    identity = torch.eye(feature_dim, dtype=compute_dtype, device=work_device)
    weights: list[torch.Tensor] = []
    history_weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for layer in range(layer_count):
        # Fit one layer at a time.  Production traces are large, and expanding
        # every layer to float64 together needlessly multiplies peak memory.
        history_parts: list[torch.Tensor] = []
        if history_trace is not None:
            layer_history = history_trace[layer]
            history_layout = torch.zeros_like(layer_history[..., 1:, :, :])
            for lag in range(1, history_length + 1):
                aligned = history_layout.clone()
                aligned[..., lag - 1 :, :, :] = layer_history[
                    ..., : frame_count - lag, :, :
                ]
                history_parts.append(aligned)

        layer_keys = feature_keys[layer].detach().to(
            device=work_device, dtype=compute_dtype
        )
        history_parts = [
            part.detach().to(device=work_device, dtype=compute_dtype)
            for part in history_parts
        ]
        y = targets[layer].detach().to(device=work_device, dtype=compute_dtype).reshape(
            -1, head_count, dim
        ).permute(1, 0, 2)

        if resolved_history_mode == "dense":
            x = torch.cat([layer_keys, *history_parts], dim=-1).reshape(
                -1, head_count, feature_dim
            ).permute(1, 0, 2)
            x_mean = x.mean(dim=1)
            y_mean = y.mean(dim=1)
            xc = x - x_mean[:, None, :]
            yc = y - y_mean[:, None, :]
            gram = torch.matmul(xc.transpose(-2, -1), xc)
            rhs = torch.matmul(xc.transpose(-2, -1), yc)
            scale = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / feature_dim
            penalty = ridge * scale.clamp_min(torch.finfo(compute_dtype).eps)
            system = gram + penalty[..., None, None] * identity
            if ridge > 0.0:
                combined_weight = torch.linalg.solve(system, rhs)
            else:
                combined_weight = torch.matmul(
                    torch.linalg.pinv(system, rtol=_pinv_rtol(compute_dtype)), rhs
                )
            bias = y_mean - torch.matmul(
                x_mean.unsqueeze(-2), combined_weight
            ).squeeze(-2)
            weights.append(combined_weight[:, :dim])
            history_weights.append(
                combined_weight[:, dim:]
                .reshape(head_count, history_length, dim, dim)
                .permute(1, 0, 2, 3)
            )
            biases.append(bias)
            continue

        # Every output channel sees all current-K channels, but only the same
        # channel from each V-history lag.  Form the small normal-equation
        # blocks directly to avoid materializing D copies of the K features.
        key_x = layer_keys.reshape(-1, head_count, dim).permute(1, 0, 2)
        observation_count = int(key_x.shape[1])
        if history_parts:
            history_x = torch.stack(history_parts, dim=-2).reshape(
                -1, head_count, history_length, dim
            ).permute(1, 3, 0, 2)
        else:
            history_x = key_x.new_empty(head_count, dim, observation_count, 0)

        y_by_output = y.permute(0, 2, 1)
        key_mean = key_x.mean(dim=1)
        history_mean = history_x.mean(dim=2)
        y_mean = y_by_output.mean(dim=2)
        key_centered = key_x - key_mean[:, None, :]
        history_centered = history_x - history_mean[:, :, None, :]
        y_centered = y_by_output - y_mean[:, :, None]

        key_gram = torch.matmul(key_centered.transpose(-2, -1), key_centered)
        key_rhs = torch.einsum("hni,hon->hoi", key_centered, y_centered)
        cross_gram = torch.einsum(
            "hni,honp->hoip", key_centered, history_centered
        )
        history_gram = torch.einsum(
            "honp,honq->hopq", history_centered, history_centered
        )
        history_rhs = torch.einsum(
            "honp,hon->hop", history_centered, y_centered
        )

        # Use one common relative ridge scale per head so the D x D K block
        # can be factored once and reused for every output channel.  History
        # energy is averaged over output channels because each channel sees
        # only its own P history features.
        key_energy = key_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        history_energy = history_gram.diagonal(
            dim1=-2, dim2=-1
        ).sum(dim=-1).mean(dim=-1)
        scale = (key_energy + history_energy) / feature_dim
        penalty = ridge * scale.clamp_min(torch.finfo(compute_dtype).eps)
        key_system = key_gram + penalty[:, None, None] * identity[:dim, :dim]
        history_system = history_gram + (
            penalty[:, None, None, None]
            * identity[:history_length, :history_length]
        )

        # Schur complement: solve the large K block once per head with all
        # output/history right-hand sides, then solve only P x P systems per
        # output channel and back-substitute the K weights.
        key_block_rhs = torch.cat(
            (
                key_rhs.permute(0, 2, 1),
                cross_gram.permute(0, 2, 1, 3).reshape(
                    head_count, dim, dim * history_length
                ),
            ),
            dim=-1,
        )
        if ridge > 0.0:
            key_block_solution = torch.linalg.solve(key_system, key_block_rhs)
        else:
            key_block_solution = torch.matmul(
                torch.linalg.pinv(
                    key_system, rtol=_pinv_rtol(compute_dtype)
                ),
                key_block_rhs,
            )
        inverse_key_rhs = key_block_solution[..., :dim].permute(0, 2, 1)
        inverse_cross = key_block_solution[..., dim:].reshape(
            head_count, dim, dim, history_length
        ).permute(0, 2, 1, 3)
        schur_system = history_system - torch.einsum(
            "hoip,hoiq->hopq", cross_gram, inverse_cross
        )
        schur_system = 0.5 * (schur_system + schur_system.transpose(-2, -1))
        schur_rhs = history_rhs - torch.einsum(
            "hoip,hoi->hop", cross_gram, inverse_key_rhs
        )
        fitted_history_weight = _solve(
            schur_system, schur_rhs, regularized=ridge > 0.0
        )
        fitted_key_weight = inverse_key_rhs - torch.einsum(
            "hoip,hop->hoi", inverse_cross, fitted_history_weight
        )
        bias = y_mean - torch.einsum(
            "hi,hoi->ho", key_mean, fitted_key_weight
        )
        bias = bias - torch.einsum(
            "hop,hop->ho", history_mean, fitted_history_weight
        )
        weights.append(fitted_key_weight.transpose(-2, -1))
        history_weights.append(fitted_history_weight.permute(2, 0, 1))
        biases.append(bias)

    weight = torch.stack(weights)
    value_history_weight = torch.stack(history_weights)
    bias = torch.stack(biases)

    return CrossKVParameters(
        weight=weight.to(device=keys.device, dtype=torch.float32).contiguous(),
        bias=bias.to(device=keys.device, dtype=torch.float32).contiguous(),
        stride=stride,
        layer_ids=layer_ids,
        history_weights=value_history_weight.to(
            device=keys.device, dtype=torch.float32
        ).contiguous(),
    )


@torch.no_grad()
def fit_legacy_cross_kv(
    reconstructed_keys: torch.Tensor,
    values: torch.Tensor,
    *,
    ridge: float = 0.1,
    stride: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
    weight_mode: str = "layer_shared",
    projection_source: str = "cross_kv",
) -> CrossKVParameters:
    """Fit the legacy source-to-V projection backbone in FP64.

    ``weight_mode="layer_head"`` preserves the reference fit.  The other modes
    pool across layers (``layer_shared``), heads (``head_shared``), or both
    (``shared``), then broadcast W and bias to the deployed tensor shapes.
    ``projection_source`` records whether the supplied source is reconstructed
    K or its decoder-visible temporal innovation.
    """
    work_device = (
        torch.device("cpu")
        if reconstructed_keys.device.type == "mps"
        else reconstructed_keys.device
    )
    heads = int(values.shape[-2])
    dim = int(values.shape[-1])
    identity = torch.eye(dim, dtype=torch.float64, device=work_device)
    layer_count = int(values.shape[0])

    if weight_mode != "layer_head":
        group_count = heads if weight_mode == "layer_shared" else (
            layer_count if weight_mode == "head_shared" else 1
        )
        counts = torch.zeros(
            group_count, dtype=torch.float64, device=work_device
        )
        sum_x = torch.zeros(
            group_count, dim, dtype=torch.float64, device=work_device
        )
        sum_y = torch.zeros_like(sum_x)
        xx = torch.zeros(
            group_count, dim, dim, dtype=torch.float64, device=work_device
        )
        xy = torch.zeros_like(xx)
        for layer in range(layer_count):
            key_records = _legacy_layer_records(reconstructed_keys, layer)
            value_records = _legacy_layer_records(values, layer)
            for key_record, value_record in zip(
                key_records, value_records, strict=True
            ):
                x_by_head = (
                    key_record[:, 1:]
                    .detach()
                    .to(work_device)
                    .reshape(-1, heads, dim)
                    .double()
                )
                y_by_head = (
                    value_record[:, 1:]
                    .detach()
                    .to(work_device)
                    .reshape(-1, heads, dim)
                    .double()
                )
                if weight_mode == "layer_shared":
                    observation_count = int(x_by_head.shape[0])
                    counts.add_(observation_count)
                    sum_x.add_(x_by_head.sum(dim=0))
                    sum_y.add_(y_by_head.sum(dim=0))
                    xx.add_(torch.einsum("nhd,nhe->hde", x_by_head, x_by_head))
                    xy.add_(torch.einsum("nhd,nhe->hde", x_by_head, y_by_head))
                    continue

                x = x_by_head.reshape(-1, dim)
                y = y_by_head.reshape(-1, dim)
                group = layer if weight_mode == "head_shared" else 0
                counts[group] += int(x.shape[0])
                sum_x[group].add_(x.sum(dim=0))
                sum_y[group].add_(y.sum(dim=0))
                xx[group].add_(x.transpose(0, 1) @ x)
                xy[group].add_(x.transpose(0, 1) @ y)
        mean_x = sum_x / counts[:, None]
        mean_y = sum_y / counts[:, None]
        gram = xx - torch.einsum(
            "gd,ge->gde", sum_x, sum_x
        ) / counts[:, None, None]
        rhs = xy - torch.einsum(
            "gd,ge->gde", sum_x, sum_y
        ) / counts[:, None, None]
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1) / dim
        penalty = ridge * scale.clamp_min(1e-12)
        matrix = torch.linalg.solve(
            gram + penalty[:, None, None] * identity, rhs
        )
        bias = mean_y - torch.einsum("gd,gde->ge", mean_x, matrix)
        if weight_mode == "layer_shared":
            weight = matrix.float().unsqueeze(0).expand(
                layer_count, heads, dim, dim
            )
            expanded_bias = bias.float().unsqueeze(0).expand(
                layer_count, heads, dim
            )
        elif weight_mode == "head_shared":
            weight = matrix.float().unsqueeze(1).expand(
                layer_count, heads, dim, dim
            )
            expanded_bias = bias.float().unsqueeze(1).expand(
                layer_count, heads, dim
            )
        else:
            weight = matrix.float().view(1, 1, dim, dim).expand(
                layer_count, heads, dim, dim
            )
            expanded_bias = bias.float().view(1, 1, dim).expand(
                layer_count, heads, dim
            )
        return CrossKVParameters(
            weight=weight.to(values.device).contiguous(),
            bias=expanded_bias.to(values.device).contiguous(),
            stride=stride,
            layer_ids=layer_ids,
            projection_source=projection_source,
        )

    all_weights: list[torch.Tensor] = []
    all_biases: list[torch.Tensor] = []
    for layer in range(layer_count):
        count = 0
        sum_x = torch.zeros(
            heads, dim, dtype=torch.float64, device=work_device
        )
        sum_y = torch.zeros_like(sum_x)
        xx = torch.zeros(
            heads, dim, dim, dtype=torch.float64, device=work_device
        )
        xy = torch.zeros_like(xx)
        key_records = _legacy_layer_records(reconstructed_keys, layer)
        value_records = _legacy_layer_records(values, layer)
        for key_record, value_record in zip(
            key_records, value_records, strict=True
        ):
            x = (
                key_record[:, 1:]
                .detach()
                .to(work_device)
                .reshape(-1, heads, dim)
                .double()
            )
            y = (
                value_record[:, 1:]
                .detach()
                .to(work_device)
                .reshape(-1, heads, dim)
                .double()
            )
            count += int(x.shape[0])
            sum_x.add_(x.sum(dim=0))
            sum_y.add_(y.sum(dim=0))
            xx.add_(torch.einsum("nhd,nhe->hde", x, x))
            xy.add_(torch.einsum("nhd,nhe->hde", x, y))
        mean_x = sum_x / count
        mean_y = sum_y / count
        gram = xx - torch.einsum(
            "hd,he->hde", sum_x, sum_x
        ) / count
        rhs = xy - torch.einsum(
            "hd,he->hde", sum_x, sum_y
        ) / count
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1) / dim
        penalty = ridge * scale.clamp_min(1e-12)
        matrix = torch.linalg.solve(
            gram + penalty[:, None, None] * identity, rhs
        )
        bias = mean_y - torch.einsum("hd,hde->he", mean_x, matrix)
        all_weights.append(matrix.float())
        all_biases.append(bias.float())

    return CrossKVParameters(
        weight=torch.stack(all_weights).to(values.device).contiguous(),
        bias=torch.stack(all_biases).to(values.device).contiguous(),
        stride=stride,
        layer_ids=layer_ids,
        projection_source=projection_source,
    )


@torch.no_grad()
def fit_temporal_layers(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    history_length: int,
    ridge: float = 0.0,
    max_gain: float = 1.0,
    stride: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> TemporalParameters:
    """Fit independent per-head temporal parameters for every layer.

    Inputs use ``[L, ..., T, H, D]`` layout.  The returned weights and bias
    have shapes ``[L,P,P,H,D]`` and ``[L,P,H,D]``.  Each layer is fitted with
    :func:`fit_temporal`, so warm-up, ridge and gain behavior are identical to
    the non-layered path.
    """
    layer_fits = [
        fit_temporal(
            reconstructed[layer],
            target[layer],
            history_length=history_length,
            shared=False,
            ridge=ridge,
            max_gain=max_gain,
            stride=stride,
        )
        for layer in range(int(reconstructed.shape[0]))
    ]
    return TemporalParameters(
        history_weights=torch.stack(
            [parameters.history_weights for parameters in layer_fits]
        ),
        history_bias=torch.stack([parameters.history_bias for parameters in layer_fits]),
        stride=stride,
        layer_ids=layer_ids,
    )


def _quantization_roundtrip(
    value: torch.Tensor,
    *,
    bits: int,
    config: CodecConfig,
    symmetric: bool,
) -> torch.Tensor:
    """Quantize/dequantize arbitrary ``[..., H, D]`` trace slices."""
    return roundtrip(
        value,
        bits=bits,
        group_size=config.group_size,
        scale_dtype=_dtype(config.scale_dtype),
        symmetric=symmetric,
    )


def _reconstruct_direct_trace(
    clean_values: torch.Tensor,
    config: CodecConfig,
    *,
    bits: int | None = None,
    residual_roundtrip: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Quantize every non-anchor trace unit without a temporal predictor."""
    residual_bits = config.k_bits if bits is None else bits
    frame_count = int(clean_values.shape[-3])
    reconstruction_dtype = _dtype(config.reconstruction_dtype)
    legacy_math = config.fit_mode == "legacy"
    rebuilt = torch.empty_like(
        clean_values,
        dtype=torch.float32 if legacy_math else reconstruction_dtype,
    )

    anchor = _anchor_roundtrip(clean_values[..., 0, :, :], config)
    rebuilt[..., 0, :, :] = (
        anchor.to(torch.bfloat16).float()
        if legacy_math
        else anchor.to(reconstruction_dtype)
    )
    for frame in range(1, frame_count):
        current = clean_values[..., frame, :, :]
        reconstructed_frame = (
            residual_roundtrip(current.float())
            if residual_roundtrip is not None
            else _quantization_roundtrip(
                current,
                bits=residual_bits,
                config=config,
                symmetric=config.residual_mode == "symmetric",
            )
        )
        rebuilt[..., frame, :, :] = (
            reconstructed_frame.to(torch.bfloat16).float()
            if legacy_math
            else reconstructed_frame.to(reconstruction_dtype)
        )
    return rebuilt


def _temporal_frame_terms(
    weight: torch.Tensor,
    bias: torch.Tensor,
    parameters: TemporalParameters,
    frame: int,
    trace: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Bias and newest-to-oldest lag coefficients for one ``[L,...,T,H,D]`` frame."""
    order = min(frame, parameters.history_length)
    prefix = (1,) * (trace.ndim - 4)
    heads, dim = int(trace.shape[-2]), int(trace.shape[-1])
    if parameters.layered:
        shape = (int(trace.shape[0]), *prefix, heads, dim)
        rows = weight[:, order - 1, :order]
        coefficients = [rows[:, lag] for lag in range(order)]
        row_bias = bias[:, order - 1]
    else:
        shape = (1, *prefix, heads, dim) if parameters.per_head else (1, *prefix, 1, dim)
        rows = weight[order - 1, :order]
        coefficients = [rows[lag] for lag in range(order)]
        row_bias = bias[order - 1]
    return row_bias.view(shape), [item.view(shape) for item in coefficients]


def _reconstruct_temporal_trace(
    clean_keys: torch.Tensor,
    parameters: TemporalParameters,
    config: CodecConfig,
    *,
    bits: int | None = None,
    residual_roundtrip: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply the codec's closed-loop anchor/residual rules to a trace.

    ``bits`` defaults to the K residual width.  A temporal V predictor runs the
    identical loop over V and passes ``config.v_bits``.
    """
    residual_bits = config.k_bits if bits is None else bits
    frame_count = int(clean_keys.shape[-3])
    if parameters.history_length == 0:
        return _reconstruct_direct_trace(
            clean_keys,
            config,
            bits=bits,
            residual_roundtrip=residual_roundtrip,
        )
    reconstruction_dtype = _dtype(config.reconstruction_dtype)
    legacy_math = config.fit_mode == "legacy"
    rebuilt = torch.empty_like(
        clean_keys,
        dtype=torch.float32 if legacy_math else reconstruction_dtype,
    )
    anchor = _anchor_roundtrip(clean_keys[..., 0, :, :], config)
    rebuilt[..., 0, :, :] = (
        anchor.to(torch.bfloat16).float()
        if legacy_math
        else anchor.to(reconstruction_dtype)
    )
    temporal_weight = parameters.history_weights.to(clean_keys.device)
    temporal_bias = parameters.history_bias.to(clean_keys.device)
    for frame in range(1, frame_count):
        bias, coefficients = _temporal_frame_terms(
            temporal_weight, temporal_bias, parameters, frame, clean_keys
        )
        prediction = torch.zeros_like(
            clean_keys[..., frame, :, :], dtype=torch.float32
        )
        for lag, coefficient in enumerate(coefficients, start=1):
            if legacy_math:
                prediction = prediction + (
                    rebuilt[..., frame - lag, :, :].float() * coefficient
                )
            else:
                prediction.add_(
                    rebuilt[..., frame - lag, :, :] * coefficient
                )
        prediction = (
            prediction + bias
            if legacy_math
            else prediction.add_(bias)
        )
        residual = clean_keys[..., frame, :, :].float() - prediction
        residual_rebuilt = (
            residual_roundtrip(residual)
            if residual_roundtrip is not None
            else _quantization_roundtrip(
                residual,
                bits=residual_bits,
                config=config,
                symmetric=config.residual_mode == "symmetric",
            )
        )
        reconstructed_frame = prediction + residual_rebuilt
        rebuilt[..., frame, :, :] = (
            reconstructed_frame.to(torch.bfloat16).float()
            if legacy_math
            else reconstructed_frame.to(reconstruction_dtype)
        )
    return rebuilt


def _reconstructed_key_innovation_trace(
    reconstructed_keys: torch.Tensor,
    parameters: TemporalParameters,
    config: CodecConfig,
) -> torch.Tensor:
    """Build the BF16 K innovation that the legacy decoder can reproduce."""
    innovation = torch.zeros_like(reconstructed_keys, dtype=torch.float32)
    temporal_weight = parameters.history_weights.to(reconstructed_keys.device)
    temporal_bias = parameters.history_bias.to(reconstructed_keys.device)
    for frame in range(1, int(reconstructed_keys.shape[-3])):
        bias, coefficients = _temporal_frame_terms(
            temporal_weight, temporal_bias, parameters, frame, reconstructed_keys
        )
        prediction = torch.zeros_like(
            reconstructed_keys[..., frame, :, :], dtype=torch.float32
        )
        for lag, coefficient in enumerate(coefficients, start=1):
            prediction = prediction + (
                reconstructed_keys[..., frame - lag, :, :].float() * coefficient
            )
        prediction = prediction + bias
        innovation[..., frame, :, :] = (
            reconstructed_keys[..., frame, :, :].float() - prediction
        ).to(torch.bfloat16).float()
    return innovation


def _reconstruct_legacy_cross_kv_record(
    reconstructed_keys: torch.Tensor,
    clean_values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    history_weights: torch.Tensor,
    config: CodecConfig,
) -> torch.Tensor:
    """Replay one contiguous legacy ``[B,F,...,H,D]`` fitting record."""
    reconstructed = [
        _anchor_roundtrip(clean_values[:, 0], config).to(torch.bfloat16).float()
    ]
    with _cuda_matmul_precision(
        clean_values.device, config.cross_matmul_precision
    ):
        for frame in range(1, int(clean_values.shape[1])):
            prediction = torch.einsum(
                "...hi,hij->...hj",
                reconstructed_keys[:, frame].float(),
                weight,
            ) + bias
            for lag in range(1, min(frame, int(history_weights.shape[0])) + 1):
                prediction = prediction + (
                    history_weights[lag - 1]
                    * reconstructed[-lag].float()
                )
            residual = clean_values[:, frame].float() - prediction
            residual_rebuilt = _quantization_roundtrip(
                residual,
                bits=config.v_bits,
                config=config,
                symmetric=False,
            )
            reconstructed.append(
                (prediction + residual_rebuilt).to(torch.bfloat16).float()
            )
    return torch.stack(reconstructed, dim=1)


def _reconstruct_cross_kv_trace(
    reconstructed_keys: torch.Tensor,
    clean_values: torch.Tensor,
    parameters: CrossKVParameters,
    config: CodecConfig,
    *,
    residual_roundtrip: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Roll out V with the same anchor, history, and residual rules as the codec."""
    layer_count = int(clean_values.shape[0])
    frame_count = int(clean_values.shape[-3])
    head_count = int(clean_values.shape[-2])
    dim = int(clean_values.shape[-1])

    reconstruction_dtype = _dtype(config.reconstruction_dtype)
    weight = parameters.weight.to(clean_values.device)
    history_weights = parameters.history_weights.to(clean_values.device)
    if config.fit_mode == "legacy" and clean_values.ndim == 7:
        if residual_roundtrip is not None:
            raise ValueError(
                "custom residual roundtrips are not supported for legacy 7D records"
            )
        rebuilt = torch.empty_like(clean_values, dtype=torch.float32)
        parameter_bias = parameters.bias.to(clean_values.device)
        for layer in range(layer_count):
            key_records = _legacy_layer_records(reconstructed_keys, layer)
            value_records = _legacy_layer_records(clean_values, layer)
            alpha = (
                history_weights[layer]
                if parameters.history_length
                else history_weights.new_empty(0)
            )
            for event, (key_record, value_record) in enumerate(
                zip(key_records, value_records, strict=True)
            ):
                record = _reconstruct_legacy_cross_kv_record(
                    key_record,
                    value_record,
                    weight[layer],
                    parameter_bias[layer],
                    alpha,
                    config,
                )
                rebuilt[layer, event].copy_(record.movedim(-4, -3))
        return rebuilt

    legacy_math = config.fit_mode == "legacy"
    prefix_dimensions = clean_values.ndim - 4
    broadcast_shape = (
        layer_count,
        *(1 for _ in range(prefix_dimensions)),
        head_count,
        dim,
    )
    bias = parameters.bias.to(clean_values.device).view(broadcast_shape)
    rebuilt = torch.empty_like(
        clean_values,
        dtype=torch.float32 if legacy_math else reconstruction_dtype,
    )
    anchor = _anchor_roundtrip(clean_values[..., 0, :, :], config)
    rebuilt[..., 0, :, :] = (
        anchor.to(torch.bfloat16).float()
        if legacy_math
        else anchor.to(reconstruction_dtype)
    )
    with _cuda_matmul_precision(clean_values.device, config.cross_matmul_precision):
        for frame in range(1, frame_count):
            if legacy_math:
                prediction = torch.einsum(
                    "l...hi,lhij->l...hj",
                    reconstructed_keys[..., frame, :, :].float(),
                    weight,
                ) + bias
            else:
                prediction = torch.einsum(
                    "l...hi,lhij->l...hj",
                    reconstructed_keys[..., frame, :, :].float(),
                    weight,
                )
                prediction.add_(bias)
            order = min(frame, parameters.history_length)
            for lag in range(1, order + 1):
                if parameters.history_diagonal:
                    history_term = (
                        history_weights[:, lag - 1].view(broadcast_shape)
                        * rebuilt[..., frame - lag, :, :].float()
                    )
                    prediction = (
                        prediction + history_term
                        if legacy_math
                        else prediction.add_(history_term)
                    )
                else:
                    prediction.add_(
                        torch.einsum(
                            "l...hi,lhij->l...hj",
                            rebuilt[..., frame - lag, :, :].float(),
                            history_weights[:, lag - 1],
                        )
                    )
            residual = clean_values[..., frame, :, :].float() - prediction
            residual_rebuilt = (
                residual_roundtrip(residual)
                if residual_roundtrip is not None
                else _quantization_roundtrip(
                    residual,
                    bits=config.v_bits,
                    config=config,
                    symmetric=config.residual_mode == "symmetric",
                )
            )
            reconstructed_frame = prediction + residual_rebuilt
            rebuilt[..., frame, :, :] = (
                reconstructed_frame.to(torch.bfloat16).float()
                if legacy_math
                else reconstructed_frame.to(reconstruction_dtype)
            )
    return rebuilt


@torch.no_grad()
def fit_legacy_temporal_v_alpha(
    reconstructed_keys: torch.Tensor,
    clean_values: torch.Tensor,
    cross: CrossKVParameters,
    config: CodecConfig,
    *,
    rounds: int = 2,
    ridge: float = 0.0,
    gain_limit: float = 1.0,
    fit_history_source: str = "reconstructed",
) -> CrossKVParameters:
    """Fit the legacy diagonal ``Cross(K_t) + sum_lag alpha_lag * Vhat_(t-lag)``.

    One coefficient per lag, head and channel.  With several lags the
    coefficients are coupled, so they are solved together over the frames where
    every lag is available; a single lag keeps its closed-form ratio so the
    validated VH1 fit is untouched.
    """
    history_length = int(config.v_history_length)
    layer_count = int(clean_values.shape[0])
    heads = int(clean_values.shape[-2])
    dim = int(clean_values.shape[-1])
    alpha = torch.zeros(
        layer_count,
        history_length,
        heads,
        dim,
        dtype=torch.float32,
        device=clean_values.device,
    )
    weight = cross.weight.to(clean_values.device)
    bias = cross.bias.to(clean_values.device)

    effective_rounds = rounds if fit_history_source == "reconstructed" else 1
    for _ in range(effective_rounds):
        rollout_parameters = CrossKVParameters(
            weight=weight,
            bias=bias,
            stride=cross.stride,
            layer_ids=cross.layer_ids,
            history_weights=alpha,
            projection_source=cross.projection_source,
        )
        rebuilt_values = (
            _reconstruct_cross_kv_trace(
                reconstructed_keys,
                clean_values,
                rollout_parameters,
                config,
            )
            if fit_history_source == "reconstructed"
            else clean_values
        )
        next_alpha: list[torch.Tensor] = []
        with _cuda_matmul_precision(
            clean_values.device, config.cross_matmul_precision
        ):
            for layer in range(layer_count):
                cross_terms = torch.zeros(
                    heads,
                    dim,
                    history_length,
                    dtype=torch.float64,
                    device=clean_values.device,
                )
                gram = torch.zeros(
                    heads,
                    dim,
                    history_length,
                    history_length,
                    dtype=torch.float64,
                    device=clean_values.device,
                )
                key_records = _legacy_layer_records(reconstructed_keys, layer)
                clean_records = _legacy_layer_records(clean_values, layer)
                rebuilt_records = _legacy_layer_records(rebuilt_values, layer)
                for key_record, clean_record, rebuilt_record in zip(
                    key_records,
                    clean_records,
                    rebuilt_records,
                    strict=True,
                ):
                    frames = int(clean_record.shape[1])
                    current_cross = torch.einsum(
                        "...hi,hij->...hj",
                        key_record[:, history_length:].float(),
                        weight[layer],
                    ) + bias[layer]
                    target = (
                        clean_record[:, history_length:].float() - current_cross
                    ).double()
                    lagged = [
                        rebuilt_record[
                            :, history_length - lag : frames - lag
                        ].double()
                        for lag in range(1, history_length + 1)
                    ]
                    reduce_axes = tuple(range(target.ndim - 2))
                    for index, previous in enumerate(lagged):
                        cross_terms[..., index].add_(
                            (previous * target).sum(dim=reduce_axes)
                        )
                        for other in range(index, history_length):
                            product = (previous * lagged[other]).sum(
                                dim=reduce_axes
                            )
                            gram[..., index, other].add_(product)
                            if other != index:
                                gram[..., other, index].add_(product)
                # Ridge scales with the per-head average history energy, as
                # the single-lag fit did: average over lags, then channels.
                diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
                penalty = ridge * diagonal.mean(dim=-1).mean(
                    dim=-1, keepdim=True
                ).clamp_min(0.0)
                if history_length == 1:
                    # One lag is an independent ratio per channel; keep the
                    # closed form so the validated VH1 fit does not move.
                    denominator = gram[..., 0, 0] + penalty
                    fitted = torch.where(
                        denominator > 1e-20,
                        cross_terms[..., 0] / denominator.clamp_min(1e-20),
                        torch.zeros_like(denominator),
                    ).unsqueeze(-1)
                else:
                    identity = torch.eye(
                        history_length,
                        dtype=torch.float64,
                        device=clean_values.device,
                    )
                    system = gram + (
                        penalty.clamp_min(1e-20)[..., None, None] * identity
                    )
                    fitted = torch.linalg.solve(
                        system, cross_terms.unsqueeze(-1)
                    ).squeeze(-1)
                next_alpha.append(
                    fitted.permute(2, 0, 1).float().clamp(-gain_limit, gain_limit)
                )
        alpha = torch.stack(next_alpha).to(clean_values.device)

    return CrossKVParameters(
        weight=weight,
        bias=bias,
        stride=cross.stride,
        layer_ids=cross.layer_ids,
        history_weights=alpha.contiguous(),
        projection_source=cross.projection_source,
    )


def stamp_parameter_identity(
    temporal: TemporalParameters,
    cross: CrossKVParameters | TemporalParameters,
    *,
    signature: str,
    coordinate_system: str | None,
) -> tuple[TemporalParameters, CrossKVParameters | TemporalParameters]:
    """Give a K/V pair one shared identity so they can only be loaded together."""
    identity = hashlib.sha256()
    identity.update(signature.encode("utf-8"))
    identity.update((coordinate_system or "unspecified").encode("utf-8"))
    v_tensors = (
        (cross.history_weights, cross.history_bias)
        if isinstance(cross, TemporalParameters)
        else (cross.weight, cross.history_weights, cross.bias)
    )
    for tensor in (temporal.history_weights, temporal.history_bias, *v_tensors):
        identity.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    parameter_set_id = identity.hexdigest()
    stamped_k = TemporalParameters(
        temporal.history_weights,
        temporal.history_bias,
        temporal.stride,
        temporal.layer_ids,
        signature,
        parameter_set_id,
        coordinate_system,
    )
    if isinstance(cross, TemporalParameters):
        return stamped_k, TemporalParameters(
            cross.history_weights,
            cross.history_bias,
            cross.stride,
            cross.layer_ids,
            signature,
            parameter_set_id,
            coordinate_system,
        )
    return stamped_k, CrossKVParameters(
        cross.weight,
        cross.bias,
        cross.stride,
        cross.layer_ids,
        signature,
        parameter_set_id,
        coordinate_system,
        cross.history_weights,
        cross.projection_source,
    )


def _initial_temporal_parameters(
    history_length: int,
    dim: int,
    *,
    device: torch.device,
    stride: int | None,
    signature: str,
    layer_count: int | None = None,
    head_count: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> TemporalParameters:
    """Lag-1 persistence, where the closed-loop temporal fit starts.

    ``history_length=0`` gives the empty map a direct predictor deploys.
    Passing ``layer_count`` and ``head_count`` gives the ``[L,P,P,H,D]`` layout;
    otherwise the map is the shared ``[P,P,D]`` one.
    """
    lead = () if layer_count is None else (layer_count,)
    tail = (dim,) if head_count is None else (head_count, dim)
    weights = torch.zeros(
        *lead, history_length, history_length, *tail,
        dtype=torch.float32,
        device=device,
    )
    if history_length:
        weights.select(len(lead) + 1, 0).fill_(1.0)
    bias = torch.zeros(
        *lead, history_length, *tail, dtype=torch.float32, device=device
    )
    return TemporalParameters(weights, bias, stride, layer_ids, signature)


def _fit_closed_loop_temporal(
    target: torch.Tensor,
    parameters: TemporalParameters,
    fit: Callable[[torch.Tensor], TemporalParameters],
    *,
    config: CodecConfig,
    bits: int,
    iterations: int,
    fit_history_source: str,
) -> TemporalParameters:
    """Refit a temporal predictor on the history its own codec rollout produces."""
    reconstructed = fit_history_source == "reconstructed"
    for _ in range(iterations if reconstructed else 1):
        history = (
            _reconstruct_temporal_trace(target, parameters, config, bits=bits)
            if reconstructed
            else target
        )
        parameters = replace(fit(history), codec_signature=parameters.codec_signature)
    return parameters


@torch.no_grad()
def fit_predictors(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    config: CodecConfig | None = None,
    history_length: int | None = None,
    iterations: int = 2,
    temporal_ridge: float = 0.0,
    cross_ridge: float = 0.0,
    v_history_ridge: float = 0.0,
    max_gain: float = 1.0,
    v_history_mode: str = "dense",
    layer_ids: tuple[int, ...] | None = None,
    coordinate_system: str | None = None,
    fit_history_source: str = "reconstructed",
) -> tuple[TemporalParameters, CrossKVParameters | TemporalParameters]:
    """Fit deployable K and V predictors from reconstructed or clean history.

    Clean traces use ``[L, ..., T, H, D]`` layout.  ``fit_history_source`` is a
    fitting-only ablation: ``"reconstructed"`` runs the deployable closed loop,
    while ``"clean"`` teacher-forces K/V history and the current K used by
    CrossKV.  Prediction targets remain the current clean K/V in both modes.

    ``v_history_mode`` selects the native V-history map: ``"dense"`` fits a
    full ``D x D`` matrix per lag, so the V history can mix channels the way
    the reference joint predictor does; ``"diagonal"`` fits one coefficient
    per output channel instead.

    Under ``predictor_direction="v_to_k"`` the two tensors exchange roles: V is
    predicted from its own temporal history and K is projected from the
    reconstructed V.  The codec equations do not mention K or V anywhere, so
    the mirror is applied by handing the fitter the pair in the other order
    rather than by maintaining a second set of equations.  The returned
    ``TemporalParameters`` therefore always describe the primary tensor and the
    returned ``CrossKVParameters`` the secondary one; the direction is stamped
    into the codec signature, so a mirrored set cannot load into a forward
    runtime.
    """
    # A NaN in the trace would otherwise surface as NaN parameters.
    if not (torch.isfinite(keys).all() and torch.isfinite(values).all()):
        raise ValueError("keys and values must contain only finite values")
    config = config or CodecConfig()
    if config.mirrored:
        keys, values = values, keys
    layer_count, frame_count, head_count, dim = (
        int(keys.shape[0]), *map(int, keys.shape[-3:])
    )
    if dim % config.group_size:
        raise ValueError(f"group_size={config.group_size} must divide D={dim}")
    if history_length is None:
        history_length = (
            3 if config.k_history_length is None else config.k_history_length
        )
    elif (
        config.k_history_length is not None
        and history_length != config.k_history_length
    ):
        raise ValueError("history_length does not match config.k_history_length")
    if not 0 <= history_length <= 8:
        raise ValueError(f"history_length must be in [0, 8], got {history_length}")
    if history_length == 0 and config.k_predictor_mode != "direct":
        raise ValueError("history_length=0 requires k_predictor_mode='direct'")
    if max(history_length, config.v_history_length) >= frame_count:
        raise ValueError(
            f"K/V history lengths must be smaller than the trace frame count "
            f"T={frame_count}"
        )
    legacy = config.fit_mode == "legacy"
    if legacy:
        if (
            is_cross_kv_v_predictor(config.v_predictor_mode)
            and config.v_history_length
            and v_history_mode != "diagonal"
        ):
            raise ValueError("legacy VH1 fitting requires v_history_mode='diagonal'")
        expected_controls = {
            "iterations": (iterations, 2),
            "temporal_ridge": (temporal_ridge, 1.0),
            "cross_ridge": (cross_ridge, 0.1),
            "v_history_ridge": (v_history_ridge, 0.0),
            "max_gain": (max_gain, 1.0),
        }
        for name, (actual, expected) in expected_controls.items():
            if float(actual) != expected:
                raise ValueError(
                    f"legacy fitting requires {name}={expected}, got {actual}"
                )

    signature = config.format_signature()
    reconstructed = fit_history_source == "reconstructed"
    rounds = iterations if reconstructed else 1
    # Native parameters are always per layer and head; legacy ones are shared
    # unless the config asks for the layer-head K predictor.
    all_layer_ids = tuple(range(layer_count)) if layer_ids is None else layer_ids
    native_layout = (
        {}
        if legacy
        else {
            "layer_count": layer_count,
            "head_count": head_count,
            "layer_ids": all_layer_ids,
        }
    )

    def closed_loop(target, history, fit, *, bits, layout):
        return _fit_closed_loop_temporal(
            target,
            _initial_temporal_parameters(
                history,
                dim,
                device=target.device,
                stride=config.stride,
                signature=signature,
                **layout,
            ),
            fit,
            config=config,
            bits=bits,
            iterations=iterations,
            fit_history_source=fit_history_source,
        )

    def fit_temporal_map(target, history, *, layer_head):
        def fit(trace: torch.Tensor) -> TemporalParameters:
            if not legacy:
                return fit_temporal_layers(
                    trace,
                    target,
                    history_length=history,
                    ridge=temporal_ridge,
                    max_gain=max_gain,
                    stride=config.stride,
                    layer_ids=layer_ids,
                )
            if layer_head:
                return fit_legacy_layer_head_temporal(
                    trace,
                    target,
                    history_length=history,
                    ridge=temporal_ridge,
                    max_gain=max_gain,
                    stride=config.stride,
                    layer_ids=layer_ids,
                )
            return fit_legacy_shared_temporal(
                trace,
                target,
                history_length=history,
                ridge=temporal_ridge,
                max_gain=max_gain,
                stride=config.stride,
            )

        return fit

    k_mode = config.k_predictor_mode
    if k_mode == "direct":
        temporal = _initial_temporal_parameters(
            0, dim, device=keys.device, stride=config.stride, signature=signature
        )
    elif is_fixed_temporal_predictor(k_mode):
        temporal = build_fixed_temporal_parameters(
            k_mode,
            history_length=history_length,
            dim=dim,
            stride=config.stride,
            device=keys.device,
            signature=signature,
            layer_ids=None if legacy else all_layer_ids,
            head_count=None if legacy else head_count,
        )
    else:
        layer_head = k_mode == "layer_head"
        temporal = closed_loop(
            keys,
            history_length,
            fit_temporal_map(keys, history_length, layer_head=layer_head),
            bits=config.k_bits,
            layout=(
                {
                    "layer_count": layer_count,
                    "head_count": head_count,
                    "layer_ids": layer_ids,
                }
                if layer_head or not legacy
                else {}
            ),
        )

    def reconstructed_keys() -> torch.Tensor:
        return (
            _reconstruct_temporal_trace(keys, temporal, config)
            if reconstructed
            else keys
        )

    v_mode = config.v_predictor_mode
    if v_mode == "direct":
        cross = _initial_temporal_parameters(
            0,
            dim,
            device=values.device,
            stride=config.stride,
            signature=signature,
            **native_layout,
        )
    elif is_fixed_temporal_predictor(v_mode):
        # The fixed family deliberately never reads the calibration trace for
        # its coefficients.  It still emits a normal TemporalParameters archive
        # so deployment and the learned temporal baseline use the same decoder.
        cross = build_fixed_temporal_parameters(
            v_mode,
            history_length=config.v_history_length,
            dim=dim,
            stride=config.stride,
            device=values.device,
            signature=signature,
            layer_ids=None if legacy else all_layer_ids,
            head_count=None if legacy else head_count,
        )
    elif v_mode in FIXED_CROSS_KV_V_PREDICTOR_MODES:
        cross = _fit_cross_kv_fixed_history(
            reconstructed_keys(),
            values,
            config=config,
            rounds=iterations,
            ridge=cross_ridge,
            layer_ids=layer_ids,
            fit_history_source=fit_history_source,
        )
    elif v_mode == "temporal":
        # Temporal V reads its own reconstructed history, so it needs neither
        # the reconstructed K nor a Cross(K) map: it is the K loop with V's
        # history length and residual width.
        cross = closed_loop(
            values,
            config.v_history_length,
            fit_temporal_map(values, config.v_history_length, layer_head=False),
            bits=config.v_bits,
            layout=(
                {}
                if legacy
                else {
                    "layer_count": layer_count,
                    "head_count": head_count,
                    "layer_ids": layer_ids,
                }
            ),
        )
    elif legacy:
        innovation = v_mode == "innovation"
        source = reconstructed_keys()
        if innovation:
            source = _reconstructed_key_innovation_trace(source, temporal, config)
        cross = fit_legacy_cross_kv(
            source,
            values,
            ridge=cross_ridge,
            stride=config.stride,
            layer_ids=layer_ids,
            weight_mode=config.cross_kv_weight_mode,
            projection_source="innovation" if innovation else "cross_kv",
        )
        if config.v_history_length:
            cross = fit_legacy_temporal_v_alpha(
                source,
                values,
                cross,
                config,
                rounds=iterations,
                ridge=v_history_ridge,
                gain_limit=max_gain,
                fit_history_source=fit_history_source,
            )
    else:
        source = reconstructed_keys()
        cross = fit_cross_kv(
            source[..., 1:, :, :],
            values[..., 1:, :, :],
            ridge=cross_ridge,
            stride=config.stride,
            layer_ids=layer_ids,
        )
        if config.v_history_length:
            history = values
            if reconstructed:
                history = _reconstruct_cross_kv_trace(
                    source,
                    values,
                    replace(
                        cross,
                        history_weights=torch.zeros(
                            layer_count,
                            config.v_history_length,
                            head_count,
                            dim,
                            device=keys.device,
                            dtype=torch.float32,
                        ),
                    ),
                    config,
                )
            for round_index in range(rounds):
                if round_index:
                    history = _reconstruct_cross_kv_trace(source, values, cross, config)
                cross = fit_cross_kv(
                    source,
                    values,
                    reconstructed_values=history,
                    history_length=config.v_history_length,
                    history_mode=v_history_mode,
                    ridge=cross_ridge,
                    stride=config.stride,
                    layer_ids=layer_ids,
                )
    return stamp_parameter_identity(
        temporal, cross, signature=signature, coordinate_system=coordinate_system
    )


__all__ = [
    "build_fixed_temporal_parameters",
    "fit_cross_kv",
    "fit_predictors",
    "fit_temporal",
    "fit_temporal_layers",
]
