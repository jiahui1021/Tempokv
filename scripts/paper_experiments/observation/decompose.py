#!/usr/bin/env python3
"""Tier-ladder decomposition of KV-cache predictability (paper section 3.2).

The ladder separates what each modelling step actually buys on held-out
prompts:

    tier 0  direct quantization          prediction 0
    tier 1  per-channel DC removal       prediction mu_d
    tier 2  + diagonal temporal AR       mu + sum_i A_i (X_{t-i} - mu)
    tier 3  + aligned K (V only)         tier 2 + W_{K->V} K_t

Every tier is evaluated on the same rows.  Different lag orders would
otherwise drop different numbers of leading frames, change the denominator,
and make the ladder non-additive; a single warm-up ``p`` shared by all tiers
is what keeps tier deltas readable as "what this term earned".

Energy is not bits.  The quantizer's step size follows the per-group range,
and the range follows the outliers, so ``bits_saved`` (a range ratio) is
reported next to ``energy_equivalent_bits`` (``0.5 log2(1/energy_ratio)``).
The gap between them is a result, not a bug.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterator

import torch


__all__ = [
    "TierStats",
    "ar_step",
    "build_lags",
    "channel_mean",
    "closed_loop_rollout",
    "cross_step",
    "fit_ar",
    "fit_cross",
    "group_range",
    "predict_ar",
    "predict_cross",
    "residual_stats",
    "roundtrip_slice",
]


# ---------------------------------------------------------------------------
# lag construction
# ---------------------------------------------------------------------------


def channel_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-channel DC term of ``[S, T, D]`` (or any ``[..., D]``) samples."""
    if x.ndim < 2:
        raise ValueError(f"expected at least a [N, D] tensor, got {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1]).double().mean(0).to(x.dtype)


def build_lags(
    x: torch.Tensor,
    order: int,
    *,
    warmup: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[S, T, D]`` -> lags ``[N, order, D]`` and targets ``[N, D]``.

    ``warmup`` is the number of leading frames dropped and defaults to
    ``order``.  Passing a larger value lines a low-order predictor up with the
    rows a higher-order one can reach, which is what makes the tiers
    comparable.  Lags are built along the frame axis only, so they never cross
    a sample (and therefore never cross a prompt) boundary.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [S, T, D] samples, got {tuple(x.shape)}")
    if order < 0:
        raise ValueError("order must not be negative")
    samples, frames, dim = x.shape
    warmup = order if warmup is None else warmup
    if warmup < order:
        raise ValueError("warmup must be at least the lag order")
    if warmup >= frames:
        raise ValueError(
            f"trace has {frames} frames, which leaves nothing after a "
            f"warm-up of {warmup}"
        )
    target = x[:, warmup:].reshape(-1, dim)
    if order == 0:
        return x.new_zeros(target.shape[0], 0, dim), target
    lags = torch.stack(
        [x[:, warmup - i : frames - i] for i in range(1, order + 1)], dim=-2
    )
    return lags.reshape(-1, order, dim), target


def _sample_chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, total, size):
        yield start, min(start + size, total)


# ---------------------------------------------------------------------------
# predictor fitting
# ---------------------------------------------------------------------------


def _ridge(gram: torch.Tensor, rhs: torch.Tensor, ridge: float) -> torch.Tensor:
    if ridge < 0:
        raise ValueError("ridge must not be negative")
    width = gram.shape[-1]
    regularized = gram + ridge * torch.eye(
        width, dtype=gram.dtype, device=gram.device
    )
    try:
        return torch.linalg.solve(regularized, rhs.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        # A rank-deficient normal matrix means the calibration set does not
        # constrain every coefficient.  Least squares still returns the
        # minimum-norm answer instead of failing the whole sweep.
        return torch.linalg.lstsq(regularized, rhs.unsqueeze(-1)).solution.squeeze(-1)


def fit_ar(
    centered: torch.Tensor,
    order: int,
    *,
    warmup: int | None = None,
    ridge: float = 1e-2,
    chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel diagonal AR with a bias, fitted by ridge normal equations.

    ``centered`` is ``[S, T, D]`` with the DC term already removed.  Returns
    coefficients ``A`` of shape ``[D, order]`` and a bias ``b`` of shape
    ``[D]``.  The Gram matrix is accumulated in float64 over sample chunks:
    a production trace has hundreds of thousands of rows, and a float32
    accumulation of ``sum x^2`` over that many terms loses digits the solve
    then amplifies.
    """
    if centered.ndim != 3:
        raise ValueError(f"expected [S, T, D] samples, got {tuple(centered.shape)}")
    samples, _, dim = centered.shape
    warmup = order if warmup is None else warmup
    width = order + 1
    gram = torch.zeros(dim, width, width, dtype=torch.float64, device=centered.device)
    rhs = torch.zeros(dim, width, dtype=torch.float64, device=centered.device)
    rows = 0
    for start, stop in _sample_chunks(samples, chunk):
        lags, target = build_lags(centered[start:stop], order, warmup=warmup)
        lags = lags.double()
        target = target.double()
        count = target.shape[0]
        if order:
            gram[:, :order, :order] += torch.einsum("npd,nqd->dpq", lags, lags)
            summed = lags.sum(0).T
            gram[:, :order, order] += summed
            gram[:, order, :order] += summed
            rhs[:, :order] += torch.einsum("npd,nd->dp", lags, target)
        gram[:, order, order] += count
        rhs[:, order] += target.sum(0)
        rows += count
    if rows == 0:
        raise ValueError("calibration trace produced no fitting rows")
    weights = _ridge(gram, rhs, ridge)
    return weights[:, :order].to(centered.dtype), weights[:, order].to(centered.dtype)


def fit_cross(
    centered_values: torch.Tensor,
    centered_keys: torch.Tensor,
    order: int,
    *,
    warmup: int | None = None,
    ridge: float = 1e-2,
    chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Joint fit of the diagonal V history and the dense aligned-K term.

    ``centered_values``/``centered_keys`` are ``[S, T, D]`` with their own DC
    terms removed.  Returns ``(Av [D, order], Wk [D, D], b [D])``.

    The K block is the same for every output channel, so its Gram is
    accumulated once and broadcast into the per-channel system instead of
    being recomputed 128 times.  Fitting the cross term jointly with the
    history term - rather than on top of the tier-2 residual - is what makes
    the tier 2 -> 3 delta attributable to the K term alone.
    """
    if centered_values.shape != centered_keys.shape:
        raise ValueError("V and K traces must share a shape")
    samples, _, dim = centered_values.shape
    warmup = order if warmup is None else warmup
    width = order + dim + 1
    gram = torch.zeros(
        dim, width, width, dtype=torch.float64, device=centered_values.device
    )
    rhs = torch.zeros(dim, width, dtype=torch.float64, device=centered_values.device)
    shared = torch.zeros(
        dim + 1, dim + 1, dtype=torch.float64, device=centered_values.device
    )
    for start, stop in _sample_chunks(samples, chunk):
        lags, target = build_lags(centered_values[start:stop], order, warmup=warmup)
        current = centered_keys[start:stop, warmup:].reshape(-1, dim)
        lags = lags.double()
        target = target.double()
        count = target.shape[0]
        source = torch.cat(
            [
                current.double(),
                torch.ones(count, 1, dtype=torch.float64, device=target.device),
            ],
            dim=1,
        )
        shared += source.T @ source
        if order:
            gram[:, :order, :order] += torch.einsum("nqd,nrd->dqr", lags, lags)
            cross = torch.einsum("nqd,nz->dqz", lags, source)
            gram[:, :order, order:] += cross
            gram[:, order:, :order] += cross.transpose(1, 2)
            rhs[:, :order] += torch.einsum("nqd,nd->dq", lags, target)
        rhs[:, order:] += (source.T @ target).T
    gram[:, order:, order:] += shared.unsqueeze(0)
    weights = _ridge(gram, rhs, ridge)
    dtype = centered_values.dtype
    return (
        weights[:, :order].to(dtype),
        weights[:, order : order + dim].to(dtype),
        weights[:, -1].to(dtype),
    )


def predict_ar(
    lags: torch.Tensor, coefficients: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """``lags [N, order, D]`` with ``A [D, order]`` and ``b [D]`` -> ``[N, D]``."""
    if coefficients.shape[-1] == 0:
        return bias.expand(lags.shape[0], -1)
    return torch.einsum("npd,dp->nd", lags, coefficients) + bias


def predict_cross(
    lags: torch.Tensor,
    current_keys: torch.Tensor,
    coefficients: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Tier-3 prediction: diagonal V history plus the dense aligned-K term."""
    predicted = current_keys @ projection.T + bias
    if coefficients.shape[-1]:
        predicted = predicted + torch.einsum("npd,dp->nd", lags, coefficients)
    return predicted


# ---------------------------------------------------------------------------
# closed-loop rollout
# ---------------------------------------------------------------------------


def roundtrip_slice(
    value: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    scale_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Quantize/dequantize one ``[S, D]`` frame with the production kernel.

    The head axis the quantizer expects is a formality here: groups are cut
    along ``D`` only, so a single head per row reproduces the runtime grouping
    exactly.
    """
    from tempokv.quantization import roundtrip

    if value.ndim != 2:
        raise ValueError(f"expected a [S, D] frame, got {tuple(value.shape)}")
    reshaped = value.unsqueeze(1)
    rebuilt = roundtrip(
        reshaped,
        bits=bits,
        group_size=group_size,
        scale_dtype=scale_dtype,
        symmetric=symmetric,
    )
    return rebuilt.squeeze(1).to(value.dtype)


def closed_loop_rollout(
    target: torch.Tensor,
    predict: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    anchor_bits: int,
    residual_bits: int,
    group_size: int,
    symmetric_residual: bool,
    scale_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Replay the codec loop over ``[S, T, D]`` and return the rebuilt trace.

    ``predict(rebuilt, t)`` sees only frames the decoder already holds, which
    is the whole point: an open-loop probe that predicts from clean history
    measures a quantity the deployed system never has access to.
    """
    if target.ndim != 3:
        raise ValueError(f"expected [S, T, D] samples, got {tuple(target.shape)}")
    frames = target.shape[1]
    # NaN rather than empty: a step that reaches past the frame it is
    # predicting is a bug, and NaN makes it surface immediately instead of
    # quietly folding uninitialized memory into a reported number.
    rebuilt = torch.full_like(target, float("nan"))
    rebuilt[:, 0] = roundtrip_slice(
        target[:, 0],
        bits=anchor_bits,
        group_size=group_size,
        symmetric=True,
        scale_dtype=scale_dtype,
    )
    for frame in range(1, frames):
        predicted = predict(rebuilt, frame)
        residual = target[:, frame] - predicted
        rebuilt[:, frame] = predicted + roundtrip_slice(
            residual,
            bits=residual_bits,
            group_size=group_size,
            symmetric=symmetric_residual,
            scale_dtype=scale_dtype,
        )
    return rebuilt


def ar_step(
    coefficients: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Rollout step for tier 1 (``order == 0``) and tier 2.

    Below the lag order the frame simply has no history yet; the step falls
    back to the coefficients it can apply, which affects only frames earlier
    than the shared warm-up and so never enters a reported number.
    """
    order = int(coefficients.shape[-1])

    def step(rebuilt: torch.Tensor, frame: int) -> torch.Tensor:
        predicted = bias + mean
        for index in range(min(order, frame)):
            predicted = predicted + coefficients[:, index] * (
                rebuilt[:, frame - index - 1] - mean
            )
        return predicted

    return step


def cross_step(
    coefficients: torch.Tensor,
    projection: torch.Tensor,
    bias: torch.Tensor,
    value_mean: torch.Tensor,
    rebuilt_keys: torch.Tensor,
    key_mean: torch.Tensor,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Rollout step for tier 3: the K source is the decoder-visible K."""
    order = int(coefficients.shape[-1])

    def step(rebuilt: torch.Tensor, frame: int) -> torch.Tensor:
        predicted = (
            value_mean + bias + (rebuilt_keys[:, frame] - key_mean) @ projection.T
        )
        for index in range(min(order, frame)):
            predicted = predicted + coefficients[:, index] * (
                rebuilt[:, frame - index - 1] - value_mean
            )
        return predicted

    return step


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def group_range(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """``[N, D]`` -> ``[N, D // group_size]`` per-group max minus min."""
    if x.ndim != 2:
        raise ValueError(f"expected [N, D] rows, got {tuple(x.shape)}")
    dim = x.shape[-1]
    if group_size <= 0 or dim % group_size:
        raise ValueError("group_size must divide the channel dimension")
    grouped = x.unflatten(-1, (dim // group_size, group_size))
    return grouped.amax(-1) - grouped.amin(-1)


@dataclass(frozen=True)
class TierStats:
    """One ladder cell, on held-out rows."""

    energy_ratio: float
    energy_equivalent_bits: float
    bits_saved: float
    range_over_std: float
    rows: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _median(values: torch.Tensor) -> float:
    return float(values.reshape(-1).median().item())


def residual_stats(
    residual: torch.Tensor,
    reference: torch.Tensor,
    *,
    group_size: int = 64,
) -> TierStats:
    """Energy, equivalent bits, realised range saving, and tail shape.

    ``reference`` is the tier-0 target on the same rows, so every tier of a
    ladder shares one denominator.
    """
    if residual.shape != reference.shape:
        raise ValueError("residual and reference must share a shape")
    energy = float(residual.double().pow(2).sum().item())
    base = float(reference.double().pow(2).sum().item())
    if base <= 0.0:
        raise ValueError("reference tensor has no energy")
    ratio = energy / base
    reference_range = group_range(reference, group_size).clamp_min(1e-8)
    residual_range = group_range(residual, group_size).clamp_min(1e-8)
    grouped = residual.unflatten(-1, (residual.shape[-1] // group_size, group_size))
    tail = (residual_range / grouped.std(-1).clamp_min(1e-8))
    return TierStats(
        energy_ratio=ratio,
        energy_equivalent_bits=(
            0.5 * math.log2(1.0 / ratio) if ratio > 0.0 else float("inf")
        ),
        bits_saved=_median(torch.log2(reference_range / residual_range)),
        range_over_std=_median(tail),
        rows=int(residual.shape[0]),
    )
