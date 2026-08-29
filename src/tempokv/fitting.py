"""Fit the small linear predictors consumed by :mod:`tempokv.codec`.

The routines in this module deliberately operate on reconstructed history.
That keeps fitting consistent with decoding, where clean past frames are not
available.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import torch

from .attention_objective import (
    cross_kv_rollout,
    normalized_errors,
    objective_terms,
    phase_score,
    temporal_rollout,
)
from .codec import _cuda_matmul_precision
from .config import CodecConfig
from .parameters import CrossKVParameters, TemporalParameters
from .quantization import (
    dequantize_asymmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantize_symmetric,
    roundtrip,
)


_SCALE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _check_scalar(name: str, value: float, *, positive: bool = False) -> float:
    value = float(value)
    valid = math.isfinite(value) and (value > 0.0 if positive else value >= 0.0)
    if not valid:
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {relation}, got {value}")
    return value


def _check_stride(stride: int | None) -> int | None:
    if stride is not None and (isinstance(stride, bool) or int(stride) != stride or stride <= 0):
        raise ValueError(f"stride must be a positive integer or None, got {stride!r}")
    return None if stride is None else int(stride)


# Scanning a production trace for non-finite values costs about as much as a
# full pass over it, and the fitter re-validates the same tensors dozens of
# times as it walks layers and iterations. `fit_predictors` checks once up
# front and then suppresses the repeat scans; every public entry point still
# performs the full check on data it has not seen.
_ASSUME_FINITE = False


@contextmanager
def _assume_finite() -> Iterator[None]:
    """Skip redundant finiteness scans on already-validated tensors."""
    global _ASSUME_FINITE
    previous = _ASSUME_FINITE
    _ASSUME_FINITE = True
    try:
        yield
    finally:
        _ASSUME_FINITE = previous


def _check_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_dimensions: int,
    layout: str,
) -> None:
    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("source and target must be torch.Tensor instances")
    if source.shape != target.shape or source.ndim < minimum_dimensions:
        raise ValueError(
            f"source and target must have matching {layout} shapes, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}"
        )
    if any(size <= 0 for size in source.shape):
        raise ValueError(f"all input dimensions must be positive, got {tuple(source.shape)}")
    if not source.is_floating_point() or not target.is_floating_point():
        raise TypeError("source and target must use floating-point dtypes")
    if source.device != target.device:
        raise ValueError(
            f"source and target must be on the same device, got {source.device} and {target.device}"
        )
    if _ASSUME_FINITE:
        return
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("source and target must contain only finite values")


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
        if config.v_predictor_mode == "temporal":
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
            if value_history_weights is None:
                raise ValueError("CrossKV refinement needs V history weights")
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
            if reference_sums is None:
                raise ValueError("attention backward needs baseline reference sums")
            if backward_tensor == "key":
                own_error = terms.key_error
                own_reference = reference_sums[1]
            elif backward_tensor == "value":
                own_error = terms.value_error
                own_reference = reference_sums[3]
            else:
                raise ValueError("backward tensor must be key or value")
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
    if steps <= 0:
        raise ValueError("refinement steps must be positive")
    learning_rate = _check_scalar("learning_rate", learning_rate, positive=True)
    if patience <= 0:
        raise ValueError("refinement patience must be positive")
    if (
        not math.isfinite(min_relative_improvement)
        or min_relative_improvement < 0.0
        or min_relative_improvement >= 1.0
    ):
        raise ValueError("min_relative_improvement must be finite and in [0, 1)")
    if layer_chunk <= 0:
        raise ValueError("layer_chunk must be positive")
    max_gain = _check_scalar("max_gain", max_gain, positive=True)
    if optimizer_name not in {"adam", "sign_gradient"}:
        raise ValueError("optimizer_name must be adam or sign_gradient")
    for name, weight in (
        ("reconstruction_weight", reconstruction_weight),
        ("attention_weight", attention_weight),
    ):
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isclose(
        reconstruction_weight + attention_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("objective weights must sum to 1.0")
    phases_to_run = tuple(refine_tensors)
    if not phases_to_run or any(item not in {"key", "value"} for item in phases_to_run):
        raise ValueError("refine_tensors must contain key, value, or both")
    if len(set(phases_to_run)) != len(phases_to_run):
        raise ValueError("refine_tensors must not contain duplicates")
    if config.k_predictor_mode == "direct" and "key" in phases_to_run:
        raise ValueError("direct K has no trainable predictor for attention refinement")
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
    if not sign_step_sizes:
        raise ValueError("sign-gradient refinement needs at least one step size")
    for step_size in sign_step_sizes:
        if not math.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("sign-gradient step sizes must be positive and finite")

    if config.v_predictor_mode == "temporal" and not isinstance(
        value_parameters, TemporalParameters
    ):
        raise TypeError("temporal V refinement requires TemporalParameters")
    if config.v_predictor_mode == "cross_kv" and not isinstance(
        value_parameters, CrossKVParameters
    ):
        raise TypeError("CrossKV refinement requires CrossKVParameters")

    current: dict[str, torch.Tensor] = {
        "key_weight": key_parameters.history_weights.detach().clone(),
        "key_bias": key_parameters.history_bias.detach().clone(),
    }
    if isinstance(value_parameters, TemporalParameters):
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
        best: tuple[
            float,
            int | None,
            float | None,
            dict[str, float],
            dict[str, torch.Tensor],
        ] | None
        best = None
        steps_run = 0

        if optimizer_name == "adam":
            trainable = {
                name: current[name].clone().requires_grad_(True)
                for name in parameter_names
            }
            optimizer = torch.optim.Adam(trainable.values(), lr=learning_rate)
            stale_steps = 0
            for step in range(1, steps + 1):
                steps_run = step
                optimizer.zero_grad(set_to_none=True)
                _objective_sums(
                    keys,
                    values,
                    query_post_rope,
                    key_post_rope,
                    **{
                        **{name: current[name] for name in current},
                        **trainable,
                    },
                    config=config,
                    layer_chunk=layer_chunk,
                    backward_tensor=tensor,
                    reference_sums=baseline_sums,
                    reconstruction_weight=reconstruction_weight,
                    attention_weight=attention_weight,
                )
                if any(parameter.grad is None for parameter in trainable.values()):
                    raise ValueError(f"{tensor} refinement produced no gradient")
                if not all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in trainable.values()
                    if parameter.grad is not None
                ):
                    raise ValueError(
                        f"{tensor} refinement produced a non-finite gradient"
                    )
                optimizer.step()
                gain_name = (
                    "key_weight"
                    if tensor == "key"
                    else (
                        "value_weight"
                        if isinstance(value_parameters, TemporalParameters)
                        else (
                            "value_history_weights"
                            if "value_history_weights" in trainable
                            else None
                        )
                    )
                )
                if gain_name is not None:
                    with torch.no_grad():
                        _project_gain(
                            trainable[gain_name],
                            max_gain,
                            lag_axis=(
                                1 if gain_name == "value_history_weights" else None
                            ),
                        )
                candidate_values = {
                    name: parameter.detach().clone()
                    for name, parameter in trainable.items()
                }
                candidate = {**current, **candidate_values}
                candidate_errors = _evaluate(
                    keys,
                    values,
                    query_post_rope,
                    key_post_rope,
                    candidate,
                    config=config,
                    layer_chunk=layer_chunk,
                )
                score = phase_score(
                    candidate_errors,
                    baseline,
                    tensor=tensor,
                    reconstruction_weight=reconstruction_weight,
                    attention_weight=attention_weight,
                )
                regressed = any(
                    candidate_errors[name] > start_errors[name] + 1e-12
                    for name in guarded_metrics
                )
                best_score = start_score if best is None else best[0]
                relative_improvement = (best_score - score) / max(
                    abs(best_score), 1e-12
                )
                if not regressed and score < best_score:
                    best = (score, step, None, candidate_errors, candidate_values)
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
            trainable = {
                name: current[name].clone().requires_grad_(True)
                for name in parameter_names
            }
            _objective_sums(
                keys,
                values,
                query_post_rope,
                key_post_rope,
                **{
                    **{name: current[name] for name in current},
                    **trainable,
                },
                config=config,
                layer_chunk=layer_chunk,
                backward_tensor=tensor,
                reference_sums=baseline_sums,
                reconstruction_weight=reconstruction_weight,
                attention_weight=attention_weight,
            )
            if any(parameter.grad is None for parameter in trainable.values()):
                raise ValueError(f"{tensor} refinement produced no gradient")
            if not all(
                torch.isfinite(parameter.grad).all()
                for parameter in trainable.values()
                if parameter.grad is not None
            ):
                raise ValueError(
                    f"{tensor} refinement produced a non-finite gradient"
                )
            directions = {
                name: -parameter.grad.sign()
                for name, parameter in trainable.items()
                if parameter.grad is not None
            }
            for step_size in sign_step_sizes:
                candidate_values = {
                    name: current[name] + step_size * directions[name]
                    for name in parameter_names
                }
                gain_name = (
                    "key_weight"
                    if tensor == "key"
                    else (
                        "value_weight"
                        if isinstance(value_parameters, TemporalParameters)
                        else (
                            "value_history_weights"
                            if "value_history_weights" in candidate_values
                            else None
                        )
                    )
                )
                if gain_name is not None:
                    _project_gain(
                        candidate_values[gain_name],
                        max_gain,
                        lag_axis=(1 if gain_name == "value_history_weights" else None),
                    )
                candidate = {**current, **candidate_values}
                candidate_errors = _evaluate(
                    keys,
                    values,
                    query_post_rope,
                    key_post_rope,
                    candidate,
                    config=config,
                    layer_chunk=layer_chunk,
                )
                score = phase_score(
                    candidate_errors,
                    baseline,
                    tensor=tensor,
                    reconstruction_weight=reconstruction_weight,
                    attention_weight=attention_weight,
                )
                regressed = any(
                    candidate_errors[name] > start_errors[name] + 1e-12
                    for name in guarded_metrics
                )
                best_score = start_score if best is None else best[0]
                if not regressed and score < best_score:
                    best = (score, None, step_size, candidate_errors, candidate_values)

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
        score, best_step, step_size, errors, candidate_values = best
        current.update(candidate_values)
        phases.append(
            PhaseResult(
                tensor,
                True,
                best_step,
                step_size,
                steps_run,
                start_score,
                score,
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
        if isinstance(value_parameters, TemporalParameters)
        else replace(
            value_parameters,
            weight=current["value_weight"],
            bias=current["value_bias"],
            history_weights=current["value_history_weights"],
        )
    )
    refined_key, refined_value = _stamp_parameter_identity(
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
    _check_pair(
        reconstructed,
        target,
        minimum_dimensions=3,
        layout="[..., T, H, D]",
    )
    if isinstance(history_length, bool) or int(history_length) != history_length:
        raise ValueError(f"history_length must be a positive integer, got {history_length!r}")
    history_length = int(history_length)
    if history_length <= 0:
        raise ValueError(f"history_length must be positive, got {history_length}")
    frame_count, head_count, dim = map(int, reconstructed.shape[-3:])
    if frame_count <= history_length:
        raise ValueError(
            f"T={frame_count} must be greater than history_length={history_length}"
        )
    if not isinstance(shared, bool):
        raise TypeError(f"shared must be bool, got {type(shared).__name__}")
    ridge = _check_scalar("ridge", ridge)
    max_gain = _check_scalar("max_gain", max_gain, positive=True)
    stride = _check_stride(stride)

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
    _check_pair(
        reconstructed,
        target,
        minimum_dimensions=4,
        layout="[L, ..., T, H, D]",
    )
    if isinstance(history_length, bool) or int(history_length) != history_length:
        raise ValueError(
            f"history_length must be a positive integer, got {history_length!r}"
        )
    history_length = int(history_length)
    frame_count = int(target.shape[-3])
    dim = int(target.shape[-1])
    if history_length <= 0 or history_length >= frame_count:
        raise ValueError("history_length must be positive and smaller than T")
    ridge = _check_scalar("ridge", ridge)
    max_gain = _check_scalar("max_gain", max_gain, positive=True)
    stride = _check_stride(stride)
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
        if count <= 0:
            raise ValueError(f"no K fitting samples for history order {order}")
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
    _check_pair(keys, values, minimum_dimensions=4, layout="[L, ..., H, D]")
    if (
        isinstance(history_length, bool)
        or int(history_length) != history_length
        or not 0 <= int(history_length) <= 5
    ):
        raise ValueError("history_length must be an integer from 0 to 5")
    history_length = int(history_length)
    if history_mode not in ("dense", "diagonal"):
        raise ValueError("history_mode must be 'dense' or 'diagonal'")
    # A zero-history fit is exactly the legacy K-only regression.  Keep that
    # fast path even when callers preselect diagonal mode for a later fit.
    resolved_history_mode = "dense" if history_length == 0 else history_mode
    ridge = _check_scalar("ridge", ridge)
    stride = _check_stride(stride)
    history_trace: torch.Tensor | None = None
    feature_keys = keys
    targets = values
    if history_length:
        if reconstructed_values is None:
            raise ValueError(
                "reconstructed_values is required when history_length is positive"
            )
        history_trace = reconstructed_values
        _check_pair(
            history_trace,
            values,
            minimum_dimensions=4,
            layout="[L, ..., T, H, D]",
        )
        frame_count = int(keys.shape[-3])
        if frame_count <= history_length:
            raise ValueError(
                f"T={frame_count} must be greater than history_length={history_length}"
            )
        feature_keys = keys[..., 1:, :, :]
        targets = values[..., 1:, :, :]
    elif reconstructed_values is not None:
        raise ValueError(
            "reconstructed_values requires a positive history_length"
        )
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
    _check_pair(
        reconstructed_keys,
        values,
        minimum_dimensions=4,
        layout="[L, ..., T, H, D]",
    )
    if int(values.shape[-3]) < 2:
        raise ValueError("CrossKV fitting requires an anchor and one target frame")
    ridge = _check_scalar("ridge", ridge)
    stride = _check_stride(stride)
    supported_weight_modes = {
        "layer_head",
        "layer_shared",
        "head_shared",
        "shared",
    }
    if weight_mode not in supported_weight_modes:
        raise ValueError(
            "weight_mode must be layer_head, layer_shared, head_shared, or shared"
        )
    if projection_source not in {"cross_kv", "innovation"}:
        raise ValueError("projection_source must be cross_kv or innovation")
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
        if bool((counts <= 0).any()):
            raise ValueError(
                f"no V fitting samples for cross_kv weight mode {weight_mode}"
            )
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
        if count <= 0:
            raise ValueError(f"no V fitting samples for layer {layer}")
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
    _check_pair(
        reconstructed,
        target,
        minimum_dimensions=4,
        layout="[L, ..., T, H, D]",
    )
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
        scale_dtype=_SCALE_DTYPES[config.scale_dtype],
        symmetric=symmetric,
    )


def _reconstruct_direct_trace(
    clean_values: torch.Tensor,
    config: CodecConfig,
    *,
    bits: int | None = None,
) -> torch.Tensor:
    """Quantize every non-anchor trace unit without a temporal predictor."""
    residual_bits = config.k_bits if bits is None else bits
    frame_count = int(clean_values.shape[-3])
    reconstruction_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.reconstruction_dtype]
    legacy_math = config.fit_mode == "legacy"
    rebuilt = torch.empty_like(
        clean_values,
        dtype=torch.float32 if legacy_math else reconstruction_dtype,
    )

    anchor = _quantization_roundtrip(
        clean_values[..., 0, :, :],
        bits=config.anchor_bits,
        config=config,
        symmetric=True,
    )
    rebuilt[..., 0, :, :] = (
        anchor.to(torch.bfloat16).float()
        if legacy_math
        else anchor.to(reconstruction_dtype)
    )
    for frame in range(1, frame_count):
        reconstructed_frame = _quantization_roundtrip(
            clean_values[..., frame, :, :],
            bits=residual_bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        rebuilt[..., frame, :, :] = (
            reconstructed_frame.to(torch.bfloat16).float()
            if legacy_math
            else reconstructed_frame.to(reconstruction_dtype)
        )
    return rebuilt


def _reconstruct_temporal_trace(
    clean_keys: torch.Tensor,
    parameters: TemporalParameters,
    config: CodecConfig,
    *,
    bits: int | None = None,
) -> torch.Tensor:
    """Apply the codec's closed-loop anchor/residual rules to a trace.

    ``bits`` defaults to the K residual width.  A temporal V predictor runs the
    identical loop over V and passes ``config.v_bits``.
    """
    residual_bits = config.k_bits if bits is None else bits
    layer_count = int(clean_keys.shape[0])
    frame_count = int(clean_keys.shape[-3])
    head_count = int(clean_keys.shape[-2])
    dim = int(clean_keys.shape[-1])
    if parameters.history_length == 0:
        return _reconstruct_direct_trace(clean_keys, config, bits=bits)
    if parameters.per_head:
        if tuple(parameters.history_weights.shape[-2:]) != (head_count, dim):
            raise ValueError(
                "per-head temporal parameters do not match trace heads/head dimension"
            )
    elif int(parameters.history_weights.shape[-1]) != dim:
        raise ValueError("shared legacy temporal parameters do not match trace dimension")
    if parameters.layered and int(parameters.history_weights.shape[0]) != layer_count:
        raise ValueError("temporal parameters do not match trace layer count")

    reconstruction_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.reconstruction_dtype]
    legacy_math = config.fit_mode == "legacy"
    rebuilt = torch.empty_like(
        clean_keys,
        dtype=torch.float32 if legacy_math else reconstruction_dtype,
    )
    anchor = _quantization_roundtrip(
        clean_keys[..., 0, :, :],
        bits=config.anchor_bits,
        config=config,
        symmetric=True,
    )
    rebuilt[..., 0, :, :] = (
        anchor.to(torch.bfloat16).float()
        if legacy_math
        else anchor.to(reconstruction_dtype)
    )
    prefix_dimensions = clean_keys.ndim - 4
    layered_shape = (
        layer_count,
        *(1 for _ in range(prefix_dimensions)),
        head_count,
        dim,
    )
    per_head_shape = (
        1,
        *(1 for _ in range(prefix_dimensions)),
        head_count,
        dim,
    )
    channel_shape = (
        1,
        *(1 for _ in range(prefix_dimensions + 1)),
        dim,
    )
    temporal_weight = parameters.history_weights.to(clean_keys.device)
    temporal_bias = parameters.history_bias.to(clean_keys.device)
    for frame in range(1, frame_count):
        order = min(frame, parameters.history_length)
        if parameters.layered:
            bias = temporal_bias[:, order - 1].view(layered_shape)
            selected = temporal_weight[:, order - 1, :order]
            coefficient_shape = layered_shape
        elif parameters.per_head:
            bias = temporal_bias[order - 1].view(per_head_shape)
            selected = temporal_weight[order - 1, :order]
            coefficient_shape = per_head_shape
        else:
            bias = temporal_bias[order - 1].view(channel_shape)
            selected = temporal_weight[order - 1, :order]
            coefficient_shape = channel_shape
        prediction = torch.zeros_like(
            clean_keys[..., frame, :, :], dtype=torch.float32
        )
        for lag in range(1, order + 1):
            coefficient = (
                selected[..., lag - 1, :, :]
                if parameters.layered
                else selected[lag - 1]
            )
            coefficient = coefficient.view(coefficient_shape)
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
        residual_rebuilt = _quantization_roundtrip(
            residual,
            bits=residual_bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
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
    if config.fit_mode != "legacy":
        raise ValueError("innovation fitting only supports legacy codec math")
    if parameters.history_length < 1:
        raise ValueError("K innovation requires a temporal K predictor")
    layer_count = int(reconstructed_keys.shape[0])
    frame_count = int(reconstructed_keys.shape[-3])
    head_count = int(reconstructed_keys.shape[-2])
    dim = int(reconstructed_keys.shape[-1])
    if parameters.per_head:
        if tuple(parameters.history_weights.shape[-2:]) != (head_count, dim):
            raise ValueError(
                "per-head temporal parameters do not match trace heads/head dimension"
            )
    elif int(parameters.history_weights.shape[-1]) != dim:
        raise ValueError("shared temporal parameters do not match trace dimension")
    if parameters.layered and int(parameters.history_weights.shape[0]) != layer_count:
        raise ValueError("temporal parameters do not match trace layer count")

    innovation = torch.zeros_like(reconstructed_keys, dtype=torch.float32)
    prefix_dimensions = reconstructed_keys.ndim - 4
    layered_shape = (
        layer_count,
        *(1 for _ in range(prefix_dimensions)),
        head_count,
        dim,
    )
    per_head_shape = (
        1,
        *(1 for _ in range(prefix_dimensions)),
        head_count,
        dim,
    )
    channel_shape = (
        1,
        *(1 for _ in range(prefix_dimensions + 1)),
        dim,
    )
    temporal_weight = parameters.history_weights.to(reconstructed_keys.device)
    temporal_bias = parameters.history_bias.to(reconstructed_keys.device)
    for frame in range(1, frame_count):
        order = min(frame, parameters.history_length)
        if parameters.layered:
            bias = temporal_bias[:, order - 1].view(layered_shape)
            selected = temporal_weight[:, order - 1, :order]
            coefficient_shape = layered_shape
        elif parameters.per_head:
            bias = temporal_bias[order - 1].view(per_head_shape)
            selected = temporal_weight[order - 1, :order]
            coefficient_shape = per_head_shape
        else:
            bias = temporal_bias[order - 1].view(channel_shape)
            selected = temporal_weight[order - 1, :order]
            coefficient_shape = channel_shape
        prediction = torch.zeros_like(
            reconstructed_keys[..., frame, :, :], dtype=torch.float32
        )
        for lag in range(1, order + 1):
            coefficient = (
                selected[..., lag - 1, :, :]
                if parameters.layered
                else selected[lag - 1]
            ).view(coefficient_shape)
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
    history_weight: torch.Tensor,
    config: CodecConfig,
) -> torch.Tensor:
    """Replay one contiguous legacy ``[B,F,...,H,D]`` fitting record."""
    reconstructed = [
        _quantization_roundtrip(
            clean_values[:, 0],
            bits=config.anchor_bits,
            config=config,
            symmetric=True,
        ).to(torch.bfloat16).float()
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
            if history_weight.numel():
                prediction = (
                    prediction
                    + history_weight * reconstructed[-1].float()
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
) -> torch.Tensor:
    """Roll out V with the same anchor, history, and residual rules as the codec."""
    _check_pair(
        reconstructed_keys,
        clean_values,
        minimum_dimensions=4,
        layout="[L, ..., T, H, D]",
    )
    if not parameters.layered:
        raise ValueError("closed-loop V reconstruction requires layered parameters")
    if parameters.history_length != config.v_history_length:
        raise ValueError("V parameters do not match config.v_history_length")
    layer_count = int(clean_values.shape[0])
    frame_count = int(clean_values.shape[-3])
    head_count = int(clean_values.shape[-2])
    dim = int(clean_values.shape[-1])
    if tuple(parameters.weight.shape) != (layer_count, head_count, dim, dim):
        raise ValueError("CrossKV parameters do not match the trace layout")

    reconstruction_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.reconstruction_dtype]
    weight = parameters.weight.to(clean_values.device)
    history_weights = parameters.history_weights.to(clean_values.device)
    if config.fit_mode == "legacy" and clean_values.ndim == 7:
        rebuilt = torch.empty_like(clean_values, dtype=torch.float32)
        parameter_bias = parameters.bias.to(clean_values.device)
        for layer in range(layer_count):
            key_records = _legacy_layer_records(reconstructed_keys, layer)
            value_records = _legacy_layer_records(clean_values, layer)
            alpha = (
                history_weights[layer, 0]
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
    anchor = _quantization_roundtrip(
        clean_values[..., 0, :, :],
        bits=config.anchor_bits,
        config=config,
        symmetric=True,
    )
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
            residual_rebuilt = _quantization_roundtrip(
                residual,
                bits=config.v_bits,
                config=config,
                symmetric=config.residual_mode == "symmetric",
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
    _check_pair(
        reconstructed_keys,
        clean_values,
        minimum_dimensions=4,
        layout="[L, ..., T, H, D]",
    )
    if not cross.layered:
        raise ValueError("legacy temporal V fitting requires layered CrossKV parameters")
    history_length = int(config.v_history_length)
    if history_length < 1:
        raise ValueError("legacy temporal V history fitting requires v_history_length >= 1")
    if isinstance(rounds, bool) or int(rounds) != rounds or int(rounds) <= 0:
        raise ValueError("rounds must be a positive integer")
    rounds = int(rounds)
    if fit_history_source not in {"reconstructed", "clean"}:
        raise ValueError(
            "fit_history_source must be 'reconstructed' or 'clean'"
        )
    ridge = _check_scalar("ridge", ridge)
    gain_limit = _check_scalar("gain_limit", gain_limit)
    if gain_limit > 1.0:
        raise ValueError("gain_limit must be at most 1")
    layer_count = int(clean_values.shape[0])
    heads = int(clean_values.shape[-2])
    dim = int(clean_values.shape[-1])
    if tuple(cross.weight.shape) != (layer_count, heads, dim, dim):
        raise ValueError("CrossKV parameters do not match the trace layout")
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


def _stamp_parameter_identity(
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


@torch.no_grad()
def _fit_legacy_shared_temporal_v(
    values: torch.Tensor,
    *,
    config: CodecConfig,
    history_length: int,
    iterations: int,
    ridge: float,
    max_gain: float,
    signature: str,
    fit_history_source: str,
) -> TemporalParameters:
    """Fit V with the predictor K uses, shared across every layer and head.

    V never sees K here: the closed loop runs over V's own reconstruction, so
    the fit is the same fixed-point procedure as the K fit with V's bit width.
    """
    if history_length <= 0:
        raise ValueError("temporal V fitting requires v_history_length >= 1")
    dim = int(values.shape[-1])
    weights = torch.zeros(
        history_length, history_length, dim, dtype=torch.float32, device=values.device
    )
    weights[:, 0] = 1.0
    parameters = TemporalParameters(
        history_weights=weights,
        history_bias=torch.zeros(
            history_length, dim, dtype=torch.float32, device=values.device
        ),
        stride=config.stride,
        codec_signature=signature,
    )
    effective_iterations = iterations if fit_history_source == "reconstructed" else 1
    for _ in range(effective_iterations):
        reconstructed = (
            _reconstruct_temporal_trace(
                values, parameters, config, bits=config.v_bits
            )
            if fit_history_source == "reconstructed"
            else values
        )
        fitted = fit_legacy_shared_temporal(
            reconstructed,
            values,
            history_length=history_length,
            ridge=ridge,
            max_gain=max_gain,
            stride=config.stride,
        )
        parameters = TemporalParameters(
            fitted.history_weights,
            fitted.history_bias,
            fitted.stride,
            codec_signature=signature,
        )
    return parameters


def _fit_legacy_predictors(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    config: CodecConfig,
    history_length: int,
    iterations: int,
    temporal_ridge: float,
    cross_ridge: float,
    v_history_ridge: float,
    max_gain: float,
    layer_ids: tuple[int, ...] | None,
    coordinate_system: str | None,
    fit_history_source: str,
) -> tuple[TemporalParameters, CrossKVParameters]:
    """Reproduce the legacy shared-K/Cross+previous-V fitter."""
    dim = int(keys.shape[-1])
    signature = config.format_signature()
    direct_k = config.k_predictor_mode == "direct"
    if direct_k:
        temporal = TemporalParameters(
            history_weights=torch.empty(
                0, 0, dim, dtype=torch.float32, device=keys.device
            ),
            history_bias=torch.empty(0, dim, dtype=torch.float32, device=keys.device),
            stride=config.stride,
            codec_signature=signature,
        )
    else:
        initial_weights = torch.zeros(
            history_length,
            history_length,
            dim,
            dtype=torch.float32,
            device=keys.device,
        )
        initial_weights[:, 0] = 1.0
        temporal = TemporalParameters(
            history_weights=initial_weights,
            history_bias=torch.zeros(
                history_length,
                dim,
                dtype=torch.float32,
                device=keys.device,
            ),
            stride=config.stride,
            codec_signature=signature,
        )
    effective_iterations = iterations if fit_history_source == "reconstructed" else 1
    if direct_k:
        reconstructed_keys = (
            _reconstruct_direct_trace(keys, config)
            if fit_history_source == "reconstructed"
            else keys
        )
    else:
        for _ in range(effective_iterations):
            reconstructed_keys = (
                _reconstruct_temporal_trace(keys, temporal, config)
                if fit_history_source == "reconstructed"
                else keys
            )
            fitted = fit_legacy_shared_temporal(
                reconstructed_keys,
                keys,
                history_length=history_length,
                ridge=temporal_ridge,
                max_gain=max_gain,
                stride=config.stride,
            )
            temporal = TemporalParameters(
                fitted.history_weights,
                fitted.history_bias,
                fitted.stride,
                codec_signature=signature,
            )

    if not direct_k:
        reconstructed_keys = (
            _reconstruct_temporal_trace(keys, temporal, config)
            if fit_history_source == "reconstructed"
            else keys
        )
    if config.v_predictor_mode == "direct":
        # Pure quantization: no V predictor, mirroring the empty-history K
        # direct parameters.  The decoder reconstructs V as the quantization
        # round-trip alone (Triton direct kernel route).
        cross = TemporalParameters(
            history_weights=torch.empty(
                0, 0, dim, dtype=torch.float32, device=values.device
            ),
            history_bias=torch.empty(
                0, dim, dtype=torch.float32, device=values.device
            ),
            stride=config.stride,
            codec_signature=signature,
        )
        return _stamp_parameter_identity(
            temporal,
            cross,
            signature=signature,
            coordinate_system=coordinate_system,
        )
    if config.v_predictor_mode == "temporal":
        cross = _fit_legacy_shared_temporal_v(
            values,
            config=config,
            history_length=config.v_history_length,
            iterations=iterations,
            ridge=temporal_ridge,
            max_gain=max_gain,
            signature=signature,
            fit_history_source=fit_history_source,
        )
        return _stamp_parameter_identity(
            temporal,
            cross,
            signature=signature,
            coordinate_system=coordinate_system,
        )
    projection_source = reconstructed_keys
    if config.v_predictor_mode == "innovation":
        projection_source = _reconstructed_key_innovation_trace(
            reconstructed_keys,
            temporal,
            config,
        )
    cross = fit_legacy_cross_kv(
        projection_source,
        values,
        ridge=cross_ridge,
        stride=config.stride,
        layer_ids=layer_ids,
        weight_mode=config.cross_kv_weight_mode,
        projection_source=(
            "innovation" if config.v_predictor_mode == "innovation" else "cross_kv"
        ),
    )
    if config.v_history_length:
        cross = fit_legacy_temporal_v_alpha(
            projection_source,
            values,
            cross,
            config,
            rounds=iterations,
            ridge=v_history_ridge,
            gain_limit=max_gain,
            fit_history_source=fit_history_source,
        )
    return _stamp_parameter_identity(
        temporal,
        cross,
        signature=signature,
        coordinate_system=coordinate_system,
    )


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
) -> tuple[TemporalParameters, CrossKVParameters]:
    """Fit deployable K and V predictors from reconstructed or clean history.

    The trace is scanned for non-finite values once here; everything below
    derives from it, so the per-layer and per-iteration fits skip the repeat
    scans, which are otherwise a sizeable share of the total runtime.
    """
    _check_pair(keys, values, minimum_dimensions=4, layout="[L, ..., T, H, D]")
    with _assume_finite():
        return _fit_predictors(
            keys,
            values,
            config=config,
            history_length=history_length,
            iterations=iterations,
            temporal_ridge=temporal_ridge,
            cross_ridge=cross_ridge,
            v_history_ridge=v_history_ridge,
            max_gain=max_gain,
            v_history_mode=v_history_mode,
            layer_ids=layer_ids,
            coordinate_system=coordinate_system,
            fit_history_source=fit_history_source,
        )


@torch.no_grad()
def _fit_predictors(
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
) -> tuple[TemporalParameters, CrossKVParameters]:
    """Fit deployable K and V predictors from reconstructed or clean history.

    ``v_history_mode`` selects the V-history map: ``"dense"`` fits a full
    ``D x D`` matrix per lag, so the V history can mix channels the way the
    reference joint predictor does; ``"diagonal"`` fits one coefficient per
    output channel instead.  Diagonal is far cheaper to store but cannot
    represent cross-channel recurrence, so it is only the right choice when a
    trace shows the V history is genuinely per-channel.

    Clean traces use ``[L, ..., T, H, D]`` layout.  ``fit_history_source`` is a
    fitting-only ablation: ``"reconstructed"`` runs the deployable closed loop,
    while ``"clean"`` teacher-forces K/V history and the current K used by
    CrossKV.  Prediction targets remain the current clean K/V in both modes.
    When either maximum history setting is present, fitting pools every active
    length and returns one compact coefficient map shared by all history frames.
    """
    _check_pair(keys, values, minimum_dimensions=4, layout="[L, ..., T, H, D]")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations <= 0:
        raise ValueError(f"iterations must be a positive integer, got {iterations!r}")
    iterations = int(iterations)
    if fit_history_source not in {"reconstructed", "clean"}:
        raise ValueError(
            "fit_history_source must be 'reconstructed' or 'clean'"
        )
    resolved = config or CodecConfig()
    if not isinstance(resolved, CodecConfig):
        raise TypeError(f"config must be CodecConfig or None, got {type(resolved).__name__}")
    if int(keys.shape[-1]) % resolved.group_size:
        raise ValueError(
            f"group_size={resolved.group_size} must divide D={int(keys.shape[-1])}"
        )

    # Validate fitting controls before the first quantization pass.
    temporal_ridge = _check_scalar("temporal_ridge", temporal_ridge)
    cross_ridge = _check_scalar("cross_ridge", cross_ridge)
    v_history_ridge = _check_scalar("v_history_ridge", v_history_ridge)
    max_gain = _check_scalar("max_gain", max_gain, positive=True)
    # Checked before the first quantization pass so a typo fails immediately
    # rather than after a full closed-loop K fit.
    if v_history_mode not in ("dense", "diagonal"):
        raise ValueError("v_history_mode must be 'dense' or 'diagonal'")
    configured_k_history = resolved.k_history_length
    if history_length is None:
        history_length = 3 if configured_k_history is None else configured_k_history
    elif (
        configured_k_history is not None
        and history_length != configured_k_history
    ):
        raise ValueError(
            "history_length does not match config.k_history_length"
        )
    if isinstance(history_length, bool) or int(history_length) != history_length:
        raise ValueError(f"history_length must be a positive integer, got {history_length!r}")
    history_length = int(history_length)
    if history_length < 0 or history_length > 8:
        raise ValueError(
            "history_length must be in [0, 8], got "
            f"{history_length} for T={int(keys.shape[-3])}"
        )
    if history_length == 0:
        if resolved.k_predictor_mode != "direct":
            raise ValueError(
                "history_length=0 requires config.k_predictor_mode='direct'"
            )
    elif history_length >= int(keys.shape[-3]):
        raise ValueError(
            "history_length must be smaller than the trace frame count, got "
            f"{history_length} for T={int(keys.shape[-3])}"
        )
    if resolved.v_history_length >= int(keys.shape[-3]):
        raise ValueError(
            "config.v_history_length must be smaller than the trace frame count"
        )
    if resolved.fit_mode == "legacy":
        if (
            resolved.v_predictor_mode in {"cross_kv", "innovation"}
            and resolved.v_history_length
            and v_history_mode != "diagonal"
        ):
            raise ValueError(
                "legacy VH1 fitting requires v_history_mode='diagonal'"
            )
        expected_controls = {
            "iterations": (iterations, 2),
            "temporal_ridge": (temporal_ridge, 1.0),
            "cross_ridge": (cross_ridge, 0.1),
            "v_history_ridge": (v_history_ridge, 0.0),
            "max_gain": (max_gain, 1.0),
        }
        for name, (actual, expected) in expected_controls.items():
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    f"legacy fitting requires {name}={expected}, got {actual}"
                )
        return _fit_legacy_predictors(
            keys,
            values,
            config=resolved,
            history_length=history_length,
            iterations=iterations,
            temporal_ridge=temporal_ridge,
            cross_ridge=cross_ridge,
            v_history_ridge=v_history_ridge,
            max_gain=max_gain,
            layer_ids=layer_ids,
            coordinate_system=coordinate_system,
            fit_history_source=fit_history_source,
        )

    layer_count = int(keys.shape[0])
    head_count = int(keys.shape[-2])
    dim = int(keys.shape[-1])
    if resolved.k_predictor_mode == "direct":
        temporal = TemporalParameters(
            history_weights=torch.empty(
                0, 0, dim, dtype=torch.float32, device=keys.device
            ),
            history_bias=torch.empty(0, dim, dtype=torch.float32, device=keys.device),
            stride=resolved.stride,
            codec_signature=resolved.format_signature(),
        )
    else:
        initial_weights = torch.zeros(
            layer_count,
            history_length,
            history_length,
            head_count,
            dim,
            dtype=torch.float32,
            device=keys.device,
        )
        initial_weights[:, :, 0] = 1.0
        temporal = TemporalParameters(
            history_weights=initial_weights,
            history_bias=torch.zeros(
                layer_count,
                history_length,
                head_count,
                dim,
                dtype=torch.float32,
                device=keys.device,
            ),
            stride=resolved.stride,
            layer_ids=layer_ids,
            codec_signature=resolved.format_signature(),
        )

    effective_iterations = iterations if fit_history_source == "reconstructed" else 1
    if resolved.k_predictor_mode != "direct":
        for _ in range(effective_iterations):
            reconstructed = (
                _reconstruct_temporal_trace(keys, temporal, resolved)
                if fit_history_source == "reconstructed"
                else keys
            )
            fitted_temporal = fit_temporal_layers(
                reconstructed,
                keys,
                history_length=history_length,
                ridge=temporal_ridge,
                max_gain=max_gain,
                stride=resolved.stride,
                layer_ids=layer_ids,
            )
            temporal = TemporalParameters(
                fitted_temporal.history_weights,
                fitted_temporal.history_bias,
                fitted_temporal.stride,
                fitted_temporal.layer_ids,
                resolved.format_signature(),
            )

    if resolved.v_predictor_mode == "temporal":
        # Temporal V reads its own reconstructed V history, so it needs neither
        # the reconstructed K nor a Cross(K) map.  The loop below is the K loop
        # above with V's history length and residual width, which is why
        # fit_temporal_layers and _reconstruct_temporal_trace are both generic
        # over which cache they are given.
        v_history = resolved.v_history_length
        initial_v_weights = torch.zeros(
            layer_count,
            v_history,
            v_history,
            head_count,
            dim,
            dtype=torch.float32,
            device=values.device,
        )
        initial_v_weights[:, :, 0] = 1.0
        temporal_v = TemporalParameters(
            history_weights=initial_v_weights,
            history_bias=torch.zeros(
                layer_count,
                v_history,
                head_count,
                dim,
                dtype=torch.float32,
                device=values.device,
            ),
            stride=resolved.stride,
            layer_ids=layer_ids,
            codec_signature=resolved.format_signature(),
        )
        for _ in range(effective_iterations):
            reconstructed_values = (
                _reconstruct_temporal_trace(
                    values, temporal_v, resolved, bits=resolved.v_bits
                )
                if fit_history_source == "reconstructed"
                else values
            )
            fitted_v = fit_temporal_layers(
                reconstructed_values,
                values,
                history_length=v_history,
                ridge=temporal_ridge,
                max_gain=max_gain,
                stride=resolved.stride,
                layer_ids=layer_ids,
            )
            temporal_v = TemporalParameters(
                fitted_v.history_weights,
                fitted_v.history_bias,
                fitted_v.stride,
                fitted_v.layer_ids,
                resolved.format_signature(),
            )
        return _stamp_parameter_identity(
            temporal,
            temporal_v,
            signature=resolved.format_signature(),
            coordinate_system=coordinate_system,
        )

    reconstructed = (
        _reconstruct_temporal_trace(keys, temporal, resolved)
        if fit_history_source == "reconstructed"
        else keys
    )
    if resolved.v_history_length:
        initial_cross = fit_cross_kv(
            reconstructed[..., 1:, :, :],
            values[..., 1:, :, :],
            ridge=cross_ridge,
            stride=resolved.stride,
            layer_ids=layer_ids,
        )
        initial_joint = CrossKVParameters(
            initial_cross.weight,
            initial_cross.bias,
            initial_cross.stride,
            initial_cross.layer_ids,
            history_weights=torch.zeros(
                layer_count,
                resolved.v_history_length,
                head_count,
                dim,
                device=keys.device,
                dtype=torch.float32,
            ),
        )
        reconstructed_values = (
            _reconstruct_cross_kv_trace(
                reconstructed, values, initial_joint, resolved
            )
            if fit_history_source == "reconstructed"
            else values
        )
        for _ in range(effective_iterations):
            fitted_cross = fit_cross_kv(
                reconstructed,
                values,
                reconstructed_values=reconstructed_values,
                history_length=resolved.v_history_length,
                history_mode=v_history_mode,
                ridge=cross_ridge,
                stride=resolved.stride,
                layer_ids=layer_ids,
            )
            if fit_history_source == "reconstructed":
                reconstructed_values = _reconstruct_cross_kv_trace(
                    reconstructed, values, fitted_cross, resolved
                )
    else:
        fitted_cross = fit_cross_kv(
            reconstructed[..., 1:, :, :],
            values[..., 1:, :, :],
            ridge=cross_ridge,
            stride=resolved.stride,
            layer_ids=layer_ids,
        )

    identity = hashlib.sha256()
    identity.update(resolved.format_signature().encode("utf-8"))
    identity.update((coordinate_system or "unspecified").encode("utf-8"))
    for tensor in (
        temporal.history_weights,
        temporal.history_bias,
        fitted_cross.weight,
        fitted_cross.history_weights,
        fitted_cross.bias,
    ):
        identity.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    parameter_set_id = identity.hexdigest()
    temporal = TemporalParameters(
        temporal.history_weights,
        temporal.history_bias,
        temporal.stride,
        temporal.layer_ids,
        temporal.codec_signature,
        parameter_set_id,
        coordinate_system,
    )
    cross = CrossKVParameters(
        fitted_cross.weight,
        fitted_cross.bias,
        fitted_cross.stride,
        fitted_cross.layer_ids,
        resolved.format_signature(),
        parameter_set_id,
        coordinate_system,
        fitted_cross.history_weights,
        fitted_cross.projection_source,
    )
    return temporal, cross


# Explicit aliases are convenient at public API boundaries.
fit_temporal_parameters = fit_temporal
fit_cross_kv_parameters = fit_cross_kv
fit_closed_loop = fit_predictors


__all__ = [
    "fit_cross_kv",
    "fit_cross_kv_parameters",
    "fit_closed_loop",
    "fit_predictors",
    "fit_temporal",
    "fit_temporal_layers",
    "fit_temporal_parameters",
]
