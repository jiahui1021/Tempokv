"""Fit the small linear predictors consumed by :mod:`tempokv.codec`.

The routines in this module deliberately operate on reconstructed history.
That keeps fitting consistent with decoding, where clean past frames are not
available.
"""

from __future__ import annotations

import hashlib
import math

import torch

from .config import CodecConfig
from .parameters import CrossKVParameters, TemporalParameters
from .quantization import (
    dequantize_asymmetric,
    dequantize_symmetric,
    quantize_asymmetric,
    quantize_symmetric,
)


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
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("source and target must contain only finite values")


def _solve(system: torch.Tensor, rhs: torch.Tensor, *, regularized: bool) -> torch.Tensor:
    """Solve batched normal equations, including singular unregularized fits."""
    if regularized:
        return torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.matmul(torch.linalg.pinv(system, rtol=1e-10), rhs.unsqueeze(-1)).squeeze(-1)


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

    # Float64 normal equations make the small fits deterministic and stable.
    work_device = reconstructed.device
    compute_dtype = torch.float64
    if work_device.type == "mps":  # MPS does not implement float64 linear algebra.
        work_device = torch.device("cpu")
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


def fit_cross_kv(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    ridge: float = 0.0,
    stride: int | None = None,
    layer_ids: tuple[int, ...] | None = None,
) -> CrossKVParameters:
    """Fit independent dense ``V = K @ weight + bias`` maps per layer/head.

    Inputs use ``[L, ..., H, D]`` layout.  All dimensions between layer and
    head are pooled as observations.  The returned weight and bias have shapes
    ``[L,H,D,D]`` and ``[L,H,D]`` respectively.
    """
    _check_pair(keys, values, minimum_dimensions=4, layout="[L, ..., H, D]")
    ridge = _check_scalar("ridge", ridge)
    stride = _check_stride(stride)
    layer_count = int(keys.shape[0])
    head_count = int(keys.shape[-2])
    dim = int(keys.shape[-1])

    work_device = keys.device
    compute_dtype = torch.float64
    if work_device.type == "mps":
        work_device = torch.device("cpu")
    identity = torch.eye(dim, dtype=compute_dtype, device=work_device)
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for layer in range(layer_count):
        # Fit one layer at a time.  Production traces are large, and expanding
        # every layer to float64 together needlessly multiplies peak memory.
        x = keys[layer].detach().to(device=work_device, dtype=compute_dtype).reshape(
            -1, head_count, dim
        ).permute(1, 0, 2)
        y = values[layer].detach().to(device=work_device, dtype=compute_dtype).reshape(
            -1, head_count, dim
        ).permute(1, 0, 2)

        x_mean = x.mean(dim=1)
        y_mean = y.mean(dim=1)
        xc = x - x_mean[:, None, :]
        yc = y - y_mean[:, None, :]
        gram = torch.matmul(xc.transpose(-2, -1), xc)
        rhs = torch.matmul(xc.transpose(-2, -1), yc)
        scale = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / dim
        penalty = ridge * scale.clamp_min(torch.finfo(compute_dtype).eps)
        system = gram + penalty[..., None, None] * identity
        if ridge > 0.0:
            weight = torch.linalg.solve(system, rhs)
        else:
            weight = torch.matmul(torch.linalg.pinv(system, rtol=1e-10), rhs)
        bias = y_mean - torch.matmul(x_mean.unsqueeze(-2), weight).squeeze(-2)
        weights.append(weight)
        biases.append(bias)

    weight = torch.stack(weights)
    bias = torch.stack(biases)

    return CrossKVParameters(
        weight=weight.to(device=keys.device, dtype=torch.float32).contiguous(),
        bias=bias.to(device=keys.device, dtype=torch.float32).contiguous(),
        stride=stride,
        layer_ids=layer_ids,
    )


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
    original_shape = value.shape
    head_count, dim = map(int, original_shape[-2:])
    packed_layout = value.float().reshape(-1, head_count, 1, dim)
    scale_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.scale_dtype]
    if symmetric:
        quant, scale = quantize_symmetric(
            packed_layout, bits, config.group_size, scale_dtype
        )
        rebuilt = dequantize_symmetric(
            quant,
            scale,
            bits,
            config.group_size,
            dim,
            torch.float32,
        )
    else:
        quant, scale, zero_point = quantize_asymmetric(
            packed_layout, bits, config.group_size, scale_dtype
        )
        rebuilt = dequantize_asymmetric(
            quant,
            scale,
            zero_point,
            bits,
            config.group_size,
            dim,
            torch.float32,
        )
    return rebuilt.reshape(original_shape)


def _reconstruct_temporal_trace(
    clean_keys: torch.Tensor,
    parameters: TemporalParameters,
    config: CodecConfig,
) -> torch.Tensor:
    """Apply the codec's closed-loop anchor/residual rules to a trace."""
    if not parameters.layered:
        raise ValueError("closed-loop trace reconstruction requires layered parameters")
    layer_count = int(clean_keys.shape[0])
    frame_count = int(clean_keys.shape[-3])
    head_count = int(clean_keys.shape[-2])
    dim = int(clean_keys.shape[-1])
    if tuple(parameters.history_weights.shape[-2:]) != (head_count, dim):
        raise ValueError("temporal parameters do not match trace heads/head dimension")
    if int(parameters.history_weights.shape[0]) != layer_count:
        raise ValueError("temporal parameters do not match trace layer count")

    reconstruction_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.reconstruction_dtype]
    rebuilt = torch.empty_like(clean_keys, dtype=reconstruction_dtype)
    rebuilt[..., 0, :, :] = _quantization_roundtrip(
        clean_keys[..., 0, :, :],
        bits=config.anchor_bits,
        config=config,
        symmetric=True,
    ).to(reconstruction_dtype)
    prefix_dimensions = clean_keys.ndim - 4
    broadcast_shape = (layer_count, *(1 for _ in range(prefix_dimensions)), head_count, dim)
    for frame in range(1, frame_count):
        order = min(frame, parameters.history_length)
        prediction = parameters.history_bias[:, order - 1].to(clean_keys.device).view(
            broadcast_shape
        ).expand_as(clean_keys[..., frame, :, :]).clone()
        selected = parameters.history_weights[:, order - 1, :order].to(clean_keys.device)
        for lag in range(1, order + 1):
            coefficient = selected[:, lag - 1].view(broadcast_shape)
            prediction.add_(rebuilt[..., frame - lag, :, :] * coefficient)
        residual = clean_keys[..., frame, :, :].float() - prediction
        residual_rebuilt = _quantization_roundtrip(
            residual,
            bits=config.k_bits,
            config=config,
            symmetric=config.residual_mode == "symmetric",
        )
        rebuilt[..., frame, :, :] = (prediction + residual_rebuilt).to(
            reconstruction_dtype
        )
    return rebuilt


def fit_predictors(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    config: CodecConfig | None = None,
    history_length: int,
    iterations: int = 2,
    temporal_ridge: float = 0.0,
    cross_ridge: float = 0.0,
    max_gain: float = 1.0,
    layer_ids: tuple[int, ...] | None = None,
    coordinate_system: str | None = None,
) -> tuple[TemporalParameters, CrossKVParameters]:
    """Fit deployable K and V predictors with closed-loop K reconstruction.

    Clean traces use ``[L, ..., T, H, D]`` layout.  K starts from a persistence
    predictor.  Each iteration reconstructs K with the configured anchor and
    residual quantizers, then refits temporal parameters using that
    reconstruction as history.  CrossKV is fitted from the final reconstructed
    K, never from clean K.
    """
    _check_pair(keys, values, minimum_dimensions=4, layout="[L, ..., T, H, D]")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations <= 0:
        raise ValueError(f"iterations must be a positive integer, got {iterations!r}")
    iterations = int(iterations)
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
    max_gain = _check_scalar("max_gain", max_gain, positive=True)
    if isinstance(history_length, bool) or int(history_length) != history_length:
        raise ValueError(f"history_length must be a positive integer, got {history_length!r}")
    history_length = int(history_length)
    if history_length <= 0 or history_length >= int(keys.shape[-3]):
        raise ValueError(
            f"history_length must be in [1, T-1], got {history_length} for T={int(keys.shape[-3])}"
        )

    layer_count = int(keys.shape[0])
    head_count = int(keys.shape[-2])
    dim = int(keys.shape[-1])
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

    for _ in range(iterations):
        reconstructed = _reconstruct_temporal_trace(keys, temporal, resolved)
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

    reconstructed = _reconstruct_temporal_trace(keys, temporal, resolved)
    identity = hashlib.sha256()
    identity.update(resolved.format_signature().encode("utf-8"))
    identity.update((coordinate_system or "unspecified").encode("utf-8"))
    identity.update(temporal.history_weights.detach().cpu().contiguous().numpy().tobytes())
    identity.update(temporal.history_bias.detach().cpu().contiguous().numpy().tobytes())
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
    fitted_cross = fit_cross_kv(
        reconstructed[..., 1:, :, :],
        values[..., 1:, :, :],
        ridge=cross_ridge,
        stride=resolved.stride,
        layer_ids=layer_ids,
    )
    cross = CrossKVParameters(
        fitted_cross.weight,
        fitted_cross.bias,
        fitted_cross.stride,
        fitted_cross.layer_ids,
        resolved.format_signature(),
        parameter_set_id,
        coordinate_system,
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
