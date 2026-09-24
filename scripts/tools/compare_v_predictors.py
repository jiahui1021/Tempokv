#!/usr/bin/env python3
"""Compare temporal-only and CrossKV+V-history predictors on a held-out trace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tempokv.attention_objective import (  # noqa: E402
    apply_rotation,
    cross_kv_rollout,
    recover_rotation,
    temporal_rollout,
)
from tempokv.codec import _cuda_matmul_precision  # noqa: E402
from tempokv.config import CodecConfig, load_config  # noqa: E402
from tempokv.parameters import (  # noqa: E402
    CrossKVParameters,
    TemporalParameters,
)


EPS = 1e-12




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _view_bias(current: torch.Tensor, target_ndim: int, *, layered: bool) -> torch.Tensor:
    if layered:
        return current.view(
            current.shape[0],
            *([1] * (target_ndim - 3)),
            current.shape[-2],
            current.shape[-1],
        )
    if current.ndim == 2:
        return current.view(*([1] * (target_ndim - 2)), *current.shape)
    return current.view(*([1] * (target_ndim - 1)), current.shape[-1])


def temporal_predictions(
    target: torch.Tensor,
    rebuilt: torch.Tensor,
    parameters: TemporalParameters,
) -> torch.Tensor:
    """Return the exact pre-quantization predictions used by a temporal rollout."""
    predictions = torch.zeros_like(target)
    weights = parameters.history_weights
    bias = parameters.history_bias
    layered = parameters.layered
    for frame in range(1, target.shape[-3]):
        order = min(frame, parameters.history_length)
        current_bias = bias[:, order - 1] if layered else bias[order - 1]
        prediction = _view_bias(current_bias, target[..., frame, :, :].ndim, layered=layered)
        for lag in range(order):
            previous = rebuilt[..., frame - 1 - lag, :, :]
            current_weight = (
                weights[:, order - 1, lag] if layered else weights[order - 1, lag]
            )
            prediction = prediction + previous * _view_bias(
                current_weight, previous.ndim, layered=layered
            )
        predictions[..., frame, :, :] = prediction
    return predictions


def cross_kv_predictions(
    source: torch.Tensor,
    rebuilt: torch.Tensor,
    parameters: CrossKVParameters,
    config: CodecConfig,
) -> torch.Tensor:
    """Return the exact pre-quantization CrossKV+history predictions."""
    predictions = torch.zeros_like(rebuilt)
    layered = parameters.layered
    diagonal_ndim = 4 if layered else 3
    with _cuda_matmul_precision(source.device, config.cross_matmul_precision):
        for frame in range(1, rebuilt.shape[-3]):
            current_source = source[..., frame, :, :].float()
            if layered:
                prediction = torch.einsum(
                    "l...hi,lhij->l...hj", current_source, parameters.weight
                )
            else:
                prediction = torch.einsum(
                    "...hi,hij->...hj", current_source, parameters.weight
                )
            prediction = prediction + _view_bias(
                parameters.bias, prediction.ndim, layered=layered
            )
            order = min(frame, parameters.history_length)
            for lag in range(order):
                history_weight = (
                    parameters.history_weights[:, lag]
                    if layered
                    else parameters.history_weights[lag]
                )
                previous = rebuilt[..., frame - 1 - lag, :, :].float()
                if parameters.history_weights.ndim == diagonal_ndim:
                    contribution = previous * _view_bias(
                        history_weight, previous.ndim, layered=layered
                    )
                elif layered:
                    contribution = torch.einsum(
                        "l...hi,lhij->l...hj", previous, history_weight
                    )
                else:
                    contribution = torch.einsum(
                        "...hi,hij->...hj", previous, history_weight
                    )
                prediction = prediction + contribution
            predictions[..., frame, :, :] = prediction
    return predictions


def _flatten_samples(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1, *value.shape[-3:])


def _ratio(error: torch.Tensor, reference: torch.Tensor) -> float:
    return float(error.double().sum().item() / max(reference.double().sum().item(), EPS))


def ratio_summary(
    error: torch.Tensor,
    reference: torch.Tensor,
    *,
    prompt_count: int,
) -> dict[str, Any]:
    """Summarize squared errors shaped ``[L,N,F,H,D]``."""
    if error.shape != reference.shape or error.ndim != 5:
        raise ValueError("metric tensors must have matching [L,N,F,H,D] shapes")
    layers, samples, _, heads, _ = error.shape
    if prompt_count <= 0 or samples % prompt_count:
        raise ValueError(
            f"{samples} sampled tracks cannot be divided into {prompt_count} prompts"
        )
    error_sq = error.double().square()
    reference_sq = reference.double().square()
    tracks_per_prompt = samples // prompt_count

    layer_error = error_sq.sum(dim=(1, 2, 3, 4))
    layer_reference = reference_sq.sum(dim=(1, 2, 3, 4))
    head_error = error_sq.sum(dim=(0, 1, 2, 4))
    head_reference = reference_sq.sum(dim=(0, 1, 2, 4))
    layer_head_error = error_sq.sum(dim=(1, 2, 4))
    layer_head_reference = reference_sq.sum(dim=(1, 2, 4))
    prompt_error = error_sq.reshape(
        layers, prompt_count, tracks_per_prompt, *error.shape[2:]
    ).sum(dim=(0, 2, 3, 4, 5))
    prompt_reference = reference_sq.reshape(
        layers, prompt_count, tracks_per_prompt, *reference.shape[2:]
    ).sum(dim=(0, 2, 3, 4, 5))

    def ratios(num: torch.Tensor, den: torch.Tensor) -> list[float]:
        return (num / den.clamp_min(EPS)).cpu().tolist()

    return {
        "pooled_nmse": _ratio(error_sq, reference_sq),
        "per_prompt_nmse": ratios(prompt_error, prompt_reference),
        "per_layer_nmse": ratios(layer_error, layer_reference),
        "per_head_nmse": ratios(head_error, head_reference),
        "per_layer_head_nmse": (
            layer_head_error / layer_head_reference.clamp_min(EPS)
        ).cpu().tolist(),
        "shape": {
            "layers": layers,
            "sampled_tracks": samples,
            "tracks_per_prompt": tracks_per_prompt,
            "heads": heads,
        },
    }


@torch.no_grad()
def attention_summary(
    clean_key: torch.Tensor,
    clean_value: torch.Tensor,
    rebuilt_key: torch.Tensor,
    rebuilt_value: torch.Tensor,
    q_post_rope: torch.Tensor,
    k_post_rope: torch.Tensor,
    *,
    prompt_count: int,
) -> dict[str, Any]:
    """Recorded-Q attention NMSE without materializing all layers together."""
    tensors = [
        _flatten_samples(item)
        for item in (
            clean_key,
            clean_value,
            rebuilt_key,
            rebuilt_value,
            q_post_rope,
            k_post_rope,
        )
    ]
    clean_key, clean_value, rebuilt_key, rebuilt_value, query, clean_key_post = tensors
    cosine, sine = recover_rotation(clean_key, clean_key_post)
    rebuilt_key_post = apply_rotation(rebuilt_key, cosine, sine)
    errors: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    for layer in range(clean_key.shape[0]):
        def attn_view(item: torch.Tensor) -> torch.Tensor:
            return item[layer].permute(0, 2, 1, 3)

        reference = F.scaled_dot_product_attention(
            attn_view(query), attn_view(clean_key_post), attn_view(clean_value)
        )
        candidate = F.scaled_dot_product_attention(
            attn_view(query), attn_view(rebuilt_key_post), attn_view(rebuilt_value)
        )
        errors.append((candidate - reference).permute(0, 2, 1, 3))
        references.append(reference.permute(0, 2, 1, 3))
    return ratio_summary(
        torch.stack(errors), torch.stack(references), prompt_count=prompt_count
    )


def _parameter_count(*parameters: TemporalParameters | CrossKVParameters) -> int:
    total = 0
    for parameter in parameters:
        if isinstance(parameter, TemporalParameters):
            total += parameter.history_weights.numel() + parameter.history_bias.numel()
        else:
            total += (
                parameter.weight.numel()
                + parameter.bias.numel()
                + parameter.history_weights.numel()
            )
    return total


def _validate_parameter_pair(
    key: TemporalParameters,
    value: TemporalParameters | CrossKVParameters,
    config: CodecConfig,
) -> None:
    if not key.parameter_set_id or key.parameter_set_id != value.parameter_set_id:
        raise ValueError("K/V parameters must share a non-empty parameter_set_id")
    expected = config.format_signature()
    if key.codec_signature != expected or value.codec_signature != expected:
        raise ValueError("parameter codec signature does not match the requested config")


def _candidate(
    *,
    label: str,
    mode: str,
    keys: torch.Tensor,
    values: torch.Tensor,
    q_post_rope: torch.Tensor,
    k_post_rope: torch.Tensor,
    key_parameters: TemporalParameters,
    value_parameters: TemporalParameters | CrossKVParameters,
    config: CodecConfig,
    prompt_count: int,
) -> dict[str, Any]:
    rebuilt_key = temporal_rollout(
        keys,
        key_parameters.history_weights,
        key_parameters.history_bias,
        bits=config.k_bits,
        config=config,
        detach_residual=True,
    )
    if mode == "temporal":
        assert isinstance(value_parameters, TemporalParameters)
        rebuilt_value = temporal_rollout(
            values,
            value_parameters.history_weights,
            value_parameters.history_bias,
            bits=config.v_bits,
            config=config,
            detach_residual=True,
        )
        prediction = temporal_predictions(values, rebuilt_value, value_parameters)
    else:
        assert isinstance(value_parameters, CrossKVParameters)
        rebuilt_value = cross_kv_rollout(
            rebuilt_key,
            values,
            value_parameters.weight,
            value_parameters.bias,
            value_parameters.history_weights,
            bits=config.v_bits,
            config=config,
            detach_residual=True,
        )
        prediction = cross_kv_predictions(
            rebuilt_key, rebuilt_value, value_parameters, config
        )

    value_target = _flatten_samples(values)[..., 1:, :, :]
    prediction = _flatten_samples(prediction)[..., 1:, :, :]
    rebuilt_value_flat = _flatten_samples(rebuilt_value)[..., 1:, :, :]
    key_target = _flatten_samples(keys)[..., 1:, :, :]
    rebuilt_key_flat = _flatten_samples(rebuilt_key)[..., 1:, :, :]
    return {
        "label": label,
        "mode": mode,
        "parameter_set_id": key_parameters.parameter_set_id,
        "codec_signature": key_parameters.codec_signature,
        "parameter_count": _parameter_count(key_parameters, value_parameters),
        "pre_quant_value_residual": ratio_summary(
            prediction - value_target, value_target, prompt_count=prompt_count
        ),
        "post_quant_value_reconstruction": ratio_summary(
            rebuilt_value_flat - value_target,
            value_target,
            prompt_count=prompt_count,
        ),
        "post_quant_key_reconstruction": ratio_summary(
            rebuilt_key_flat - key_target, key_target, prompt_count=prompt_count
        ),
        "recorded_q_attention": attention_summary(
            keys,
            values,
            rebuilt_key,
            rebuilt_value,
            q_post_rope,
            k_post_rope,
            prompt_count=prompt_count,
        ),
    }


def _gain(cross: dict[str, Any], temporal: dict[str, Any], metric: str) -> float:
    cross_value = cross[metric]["pooled_nmse"]
    temporal_value = temporal[metric]["pooled_nmse"]
    return 1.0 - cross_value / max(temporal_value, EPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--temporal-params", type=Path, required=True)
    parser.add_argument("--cross-params", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.prompt_count <= 0:
        parser.error("--prompt-count must be positive")
    device = torch.device(args.device)
    base_config = load_config(args.config)
    temporal_config = replace(base_config, v_predictor_mode="temporal")
    cross_config = replace(base_config, v_predictor_mode="cross_kv")

    with np.load(args.trace, allow_pickle=False) as archive:
        missing = [
            name
            for name in ("keys", "values", "q_post_rope", "k_post_rope")
            if name not in archive
        ]
        if missing:
            parser.error("trace is missing arrays: " + ", ".join(missing))
        arrays = {
            name: torch.from_numpy(np.array(archive[name], copy=True)).to(
                device=device, dtype=torch.float32
            )
            for name in ("keys", "values", "q_post_rope", "k_post_rope")
        }
        trace_role = (
            str(np.asarray(archive["trace_role"]).item())
            if "trace_role" in archive
            else "unspecified"
        )

    temporal_key = TemporalParameters.load(
        args.temporal_params / "k_temporal.npz"
    ).to(device)
    temporal_value = TemporalParameters.load(
        args.temporal_params / "v_temporal.npz"
    ).to(device)
    cross_key = TemporalParameters.load(args.cross_params / "k_temporal.npz").to(device)
    cross_value = CrossKVParameters.load(
        args.cross_params / "v_cross_kv.npz"
    ).to(device)
    _validate_parameter_pair(temporal_key, temporal_value, temporal_config)
    _validate_parameter_pair(cross_key, cross_value, cross_config)

    temporal = _candidate(
        label="V-B temporal-only",
        mode="temporal",
        key_parameters=temporal_key,
        value_parameters=temporal_value,
        config=temporal_config,
        prompt_count=args.prompt_count,
        **arrays,
    )
    cross = _candidate(
        label="V-C CrossKV+V-history",
        mode="cross_kv",
        key_parameters=cross_key,
        value_parameters=cross_value,
        config=cross_config,
        prompt_count=args.prompt_count,
        **arrays,
    )
    metric_names = (
        "pre_quant_value_residual",
        "post_quant_value_reconstruction",
        "recorded_q_attention",
    )
    report = {
        "status": "complete",
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": _sha256(args.trace),
            "role": trace_role,
            "shape": list(arrays["keys"].shape),
            "prompt_count": args.prompt_count,
        },
        "device": str(device),
        "candidates": {"temporal": temporal, "cross_kv_history": cross},
        "cross_kv_conditional_gain": {
            name: _gain(cross, temporal, name) for name in metric_names
        },
        "gain_definition": "1 - NMSE(CrossKV+V-history) / NMSE(temporal-only)",
    }
    if not all(
        math.isfinite(value)
        for value in report["cross_kv_conditional_gain"].values()
    ):
        raise ValueError("diagnostic produced a non-finite conditional gain")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
