"""Differentiable closed-loop rollout and the attention-output objective.

Least squares fits a predictor to reproduce K and V.  What the model actually
needs is the attention output those tensors produce, and the two are not the
same target: an error the attention softmax ignores costs nothing, while a
small error in a high-attention token costs a lot.  This module supplies the
pieces to optimize the second target directly.

Everything here is shape-generic over leading batch axes and expects the
temporal layout ``[..., F, H, D]`` used by the fitting traces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import CodecConfig
from .quantization import roundtrip


EPS = 1e-12
_SCALE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _round_trip(
    value: torch.Tensor, *, bits: int, config: CodecConfig, symmetric: bool
) -> torch.Tensor:
    return roundtrip(
        value,
        bits=bits,
        group_size=config.group_size,
        scale_dtype=_SCALE_DTYPES[config.scale_dtype],
        symmetric=symmetric,
    )


def temporal_rollout(
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    bits: int,
    config: CodecConfig,
    detach_residual: bool = True,
) -> torch.Tensor:
    """Replay the codec's closed loop while staying differentiable in the parameters.

    ``weight`` is ``[P, P, D]`` and ``bias`` is ``[P, D]``, shared across every
    leading axis.  ``detach_residual`` blocks gradients through the quantized
    residual, which is the straight-through proxy the refinement relies on:
    the residual is refreshed from the current prediction each step, but only
    the prediction path carries gradient.
    """
    if target.ndim < 3:
        raise ValueError("rollout target must be [..., F, H, D]")
    if weight.ndim != 3 or weight.shape[0] != weight.shape[1]:
        raise ValueError("weight must be [P, P, D] with a square history block")
    history_length = int(weight.shape[0])
    frames = int(target.shape[-3])
    if history_length >= frames:
        raise ValueError("history length must be smaller than the frame count")

    reconstructed = [
        _round_trip(
            target[..., 0, :, :],
            bits=config.anchor_bits,
            config=config,
            symmetric=True,
        ).to(torch.bfloat16).float()
    ]
    for frame in range(1, frames):
        order = min(frame, history_length)
        prediction = bias[order - 1].view(1, -1)
        for lag in range(order):
            prediction = prediction + (
                reconstructed[-1 - lag] * weight[order - 1, lag].view(1, -1)
            )
        residual = _round_trip(
            target[..., frame, :, :] - prediction,
            bits=bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        if detach_residual:
            residual = residual.detach()
        reconstructed.append((prediction + residual).to(torch.bfloat16).float())
    return torch.stack(reconstructed, dim=-3)


def recover_rotation(
    pre_rope: torch.Tensor, post_rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the RoPE rotation the runtime applied, from a clean K pair.

    Solving for the rotation instead of recomputing it keeps the objective
    independent of every RoPE implementation detail: table dtype, frame
    offsets, and block sizes are already baked into the recorded pair.
    """
    if pre_rope.shape != post_rope.shape:
        raise ValueError("pre- and post-RoPE tensors must have the same shape")
    if int(pre_rope.shape[-1]) % 2:
        raise ValueError("RoPE recovery requires an even head dimension")
    x = pre_rope[..., 0::2].float()
    y = pre_rope[..., 1::2].float()
    u = post_rope[..., 0::2].float()
    v = post_rope[..., 1::2].float()
    denominator = (x.square() + y.square()).sum(dim=-2, keepdim=True).clamp_min(EPS)
    cosine = (x * u + y * v).sum(dim=-2, keepdim=True) / denominator
    sine = (x * v - y * u).sum(dim=-2, keepdim=True) / denominator
    magnitude = (cosine.square() + sine.square()).sqrt().clamp_min(EPS)
    return cosine / magnitude, sine / magnitude


def apply_rotation(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """Apply a recovered rotation to a candidate tensor."""
    x = value[..., 0::2].float()
    y = value[..., 1::2].float()
    rotated = torch.empty(
        value.shape, dtype=torch.float32, device=value.device
    )
    rotated[..., 0::2] = x * cosine - y * sine
    rotated[..., 1::2] = x * sine + y * cosine
    return rotated


def _attention_view(value: torch.Tensor) -> torch.Tensor:
    """Fold ``[..., F, H, D]`` into the ``[N, H, F, D]`` SDPA expects."""
    heads, dim = value.shape[-2], value.shape[-1]
    flat = value.reshape(-1, value.shape[-3], heads, dim)
    return flat.permute(0, 2, 1, 3)


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    """Squared-error sums for one batch of records, in FP64."""

    key_error: torch.Tensor
    key_reference: torch.Tensor
    value_error: torch.Tensor
    value_reference: torch.Tensor
    attention_error: torch.Tensor
    attention_reference: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        return (
            self.key_error,
            self.key_reference,
            self.value_error,
            self.value_reference,
            self.attention_error,
            self.attention_reference,
        )


def objective_terms(
    *,
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    rebuilt_key: torch.Tensor,
    rebuilt_value: torch.Tensor,
    query_post_rope: torch.Tensor,
    key_post_rope: torch.Tensor,
) -> ObjectiveTerms:
    """Reconstruction and attention-output error, both against the clean pass.

    The anchor frame is excluded: it is quantized directly and no predictor
    influences it, so including it would dilute every ratio by a constant.
    """
    key_target = clean_key[..., 1:, :, :]
    value_target = clean_value[..., 1:, :, :]
    key_error = (rebuilt_key[..., 1:, :, :] - key_target).square().sum()
    value_error = (rebuilt_value[..., 1:, :, :] - value_target).square().sum()

    cosine, sine = recover_rotation(clean_key, key_post_rope)
    rebuilt_post_rope = apply_rotation(rebuilt_key, cosine, sine)

    query = _attention_view(query_post_rope)
    with torch.no_grad():
        reference = F.scaled_dot_product_attention(
            query, _attention_view(key_post_rope), _attention_view(clean_value)
        )
    candidate = F.scaled_dot_product_attention(
        query, _attention_view(rebuilt_post_rope), _attention_view(rebuilt_value)
    )
    return ObjectiveTerms(
        key_error=key_error,
        key_reference=key_target.square().sum().detach(),
        value_error=value_error,
        value_reference=value_target.square().sum().detach(),
        attention_error=(candidate - reference).square().sum(),
        attention_reference=reference.square().sum().detach(),
    )


def normalized_errors(sums: torch.Tensor) -> dict[str, float]:
    """Turn accumulated squared-error sums into comparable NMSE values."""
    if tuple(sums.shape) != (6,):
        raise ValueError("expected the six accumulated objective sums")
    values = sums.detach().double().tolist()
    return {
        "key_nmse": values[0] / max(values[1], EPS),
        "value_nmse": values[2] / max(values[3], EPS),
        "attention_nmse": values[4] / max(values[5], EPS),
    }


def phase_score(
    errors: dict[str, float], baseline: dict[str, float], *, tensor: str
) -> float:
    """Score a candidate against the least-squares baseline.

    Half the score is the tensor the phase optimizes and half is the attention
    output, so a phase can only win by improving its own reconstruction without
    spending more attention error than it gains.
    """
    metric = f"{tensor}_nmse"
    if metric not in {"key_nmse", "value_nmse"}:
        raise ValueError(f"phase tensor must be key or value, got {tensor!r}")
    return 0.5 * errors[metric] / max(baseline[metric], EPS) + 0.5 * errors[
        "attention_nmse"
    ] / max(baseline["attention_nmse"], EPS)
