"""Differentiable closed-loop rollout and the attention-output objective.

Tensors use the ``[..., F, H, D]`` trace layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import CodecConfig
from .codec import _cuda_matmul_precision, _dtype, anchor_roundtrip
from .quantization import roundtrip


EPS = 1e-12


def _reconstruction_dtype(config: CodecConfig) -> torch.dtype:
    """The dtype the codec stores reconstructions in."""
    return _dtype(config.reconstruction_dtype)



def _round_trip(
    value: torch.Tensor, *, bits: int, config: CodecConfig, symmetric: bool
) -> torch.Tensor:
    return roundtrip(
        value,
        bits=bits,
        group_size=config.group_size,
        scale_dtype=_dtype(config.scale_dtype),
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
    """Replay the temporal codec loop, differentiable in the parameters.

    Weights are ``[P,P,D]``, ``[P,P,H,D]`` or ``[L,P,P,H,D]``.
    ``detach_residual`` makes the quantized residual a straight-through constant.
    """
    layered = weight.ndim == 5
    history_length = int(weight.shape[1 if layered else 0])
    frames = int(target.shape[-3])
    if history_length == 0:
        # Direct: an empty map, so every unit is quantized without prediction.
        reconstructed = [
            anchor_roundtrip(target[..., 0, :, :], config)
            .to(_reconstruction_dtype(config))
            .float()
        ]
        for frame in range(1, frames):
            reconstructed.append(
                _round_trip(
                    target[..., frame, :, :],
                    bits=bits,
                    config=config,
                    symmetric=config.residual_mode == "symmetric",
                ).to(_reconstruction_dtype(config)).float()
            )
        return torch.stack(reconstructed, dim=-3)

    reconstructed = [
        anchor_roundtrip(target[..., 0, :, :], config)
        .to(_reconstruction_dtype(config))
        .float()
    ]
    for frame in range(1, frames):
        order = min(frame, history_length)
        current = target[..., frame, :, :]
        if layered:
            current_bias = bias[:, order - 1]
            prediction = current_bias.view(
                current_bias.shape[0],
                *([1] * (current.ndim - 3)),
                current_bias.shape[-2],
                current_bias.shape[-1],
            )
        elif weight.ndim == 4:
            current_bias = bias[order - 1]
            prediction = current_bias.view(
                *([1] * (current.ndim - 2)),
                current_bias.shape[-2],
                current_bias.shape[-1],
            )
        else:
            current_bias = bias[order - 1]
            prediction = current_bias.view(
                *([1] * (current.ndim - 1)), current_bias.shape[-1]
            )
        for lag in range(order):
            previous = reconstructed[-1 - lag]
            if layered:
                current_weight = weight[:, order - 1, lag]
                multiplier = current_weight.view(
                    current_weight.shape[0],
                    *([1] * (previous.ndim - 3)),
                    current_weight.shape[-2],
                    current_weight.shape[-1],
                )
            elif weight.ndim == 4:
                current_weight = weight[order - 1, lag]
                multiplier = current_weight.view(
                    *([1] * (previous.ndim - 2)),
                    current_weight.shape[-2],
                    current_weight.shape[-1],
                )
            else:
                current_weight = weight[order - 1, lag]
                multiplier = current_weight.view(
                    *([1] * (previous.ndim - 1)), current_weight.shape[-1]
                )
            prediction = prediction + previous * multiplier
        residual = _round_trip(
            target[..., frame, :, :] - prediction,
            bits=bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        if detach_residual:
            residual = residual.detach()
        reconstructed.append((prediction + residual).to(_reconstruction_dtype(config)).float())
    return torch.stack(reconstructed, dim=-3)


def cross_kv_rollout(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    history_weights: torch.Tensor,
    *,
    bits: int,
    config: CodecConfig,
    detach_residual: bool = True,
) -> torch.Tensor:
    """Replay the CrossKV + V-history codec loop differentiably (K source, V target)."""
    layered = weight.ndim == 4
    diagonal_history_ndim = 4 if layered else 3
    history_length = int(history_weights.shape[1 if layered else 0])
    frames = int(target.shape[-3])

    reconstructed = [
        anchor_roundtrip(target[..., 0, :, :], config)
        .to(_reconstruction_dtype(config))
        .float()
    ]
    for frame in range(1, frames):
        current_source = source[..., frame, :, :].float()
        with _cuda_matmul_precision(source.device, config.cross_matmul_precision):
            if layered:
                prediction = torch.einsum(
                    "l...hi,lhij->l...hj", current_source, weight
                )
                prediction = prediction + bias.view(
                    bias.shape[0],
                    *([1] * (prediction.ndim - 3)),
                    bias.shape[-2],
                    bias.shape[-1],
                )
            else:
                prediction = torch.einsum("...hi,hij->...hj", current_source, weight)
                prediction = prediction + bias.view(
                    *([1] * (prediction.ndim - 2)),
                    bias.shape[-2],
                    bias.shape[-1],
                )

            order = min(frame, history_length)
            for lag in range(order):
                history_weight = (
                    history_weights[:, lag] if layered else history_weights[lag]
                )
                previous = reconstructed[-1 - lag]
                if history_weights.ndim == diagonal_history_ndim:
                    if layered:
                        contribution = previous * history_weight.view(
                            history_weight.shape[0],
                            *([1] * (previous.ndim - 3)),
                            history_weight.shape[-2],
                            history_weight.shape[-1],
                        )
                    else:
                        contribution = previous * history_weight.view(
                            *([1] * (previous.ndim - 2)),
                            history_weight.shape[-2],
                            history_weight.shape[-1],
                        )
                elif layered:
                    contribution = torch.einsum(
                        "l...hi,lhij->l...hj", previous.float(), history_weight
                    )
                else:
                    contribution = torch.einsum(
                        "...hi,hij->...hj", previous.float(), history_weight
                    )
                prediction = prediction + contribution

        residual = _round_trip(
            target[..., frame, :, :] - prediction,
            bits=bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        if detach_residual:
            residual = residual.detach()
        reconstructed.append((prediction + residual).to(_reconstruction_dtype(config)).float())
    return torch.stack(reconstructed, dim=-3)


def recover_rotation(
    pre_rope: torch.Tensor, post_rope: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the RoPE rotation applied at runtime from a pre/post-RoPE K pair."""
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
    """Reconstruction and attention-output errors, excluding the anchor frame."""
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
    values = sums.detach().double().tolist()
    return {
        "key_nmse": values[0] / max(values[1], EPS),
        "value_nmse": values[2] / max(values[3], EPS),
        "attention_nmse": values[4] / max(values[5], EPS),
    }


def phase_score(
    errors: dict[str, float],
    baseline: dict[str, float],
    *,
    tensor: str,
    reconstruction_weight: float = 0.5,
    attention_weight: float = 0.5,
) -> float:
    """Score a candidate against the least-squares baseline with the objective weights."""
    metric = f"{tensor}_nmse"
    return reconstruction_weight * errors[metric] / max(
        baseline[metric], EPS
    ) + attention_weight * errors["attention_nmse"] / max(
        baseline["attention_nmse"], EPS
    )
