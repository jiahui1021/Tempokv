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
        or int(history_length) not in (0, 1, 2)
    ):
        raise ValueError("history_length must be 0, 1, or 2")
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

    work_device = keys.device
    compute_dtype = torch.float64
    if work_device.type == "mps":
        work_device = torch.device("cpu")
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
                    torch.linalg.pinv(system, rtol=1e-10), rhs
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
                torch.linalg.pinv(key_system, rtol=1e-10), key_block_rhs
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
    prefix_dimensions = clean_values.ndim - 4
    broadcast_shape = (
        layer_count,
        *(1 for _ in range(prefix_dimensions)),
        head_count,
        dim,
    )
    bias = parameters.bias.to(clean_values.device).view(broadcast_shape)
    rebuilt = torch.empty_like(clean_values, dtype=reconstruction_dtype)
    rebuilt[..., 0, :, :] = _quantization_roundtrip(
        clean_values[..., 0, :, :],
        bits=config.anchor_bits,
        config=config,
        symmetric=True,
    ).to(reconstruction_dtype)
    for frame in range(1, frame_count):
        prediction = torch.einsum(
            "l...hi,lhij->l...hj",
            reconstructed_keys[..., frame, :, :].float(),
            weight,
        )
        order = min(frame, parameters.history_length)
        for lag in range(1, order + 1):
            if parameters.history_diagonal:
                prediction.add_(
                    rebuilt[..., frame - lag, :, :].float()
                    * history_weights[:, lag - 1].view(broadcast_shape)
                )
            else:
                prediction.add_(
                    torch.einsum(
                        "l...hi,lhij->l...hj",
                        rebuilt[..., frame - lag, :, :].float(),
                        history_weights[:, lag - 1],
                    )
                )
        prediction.add_(bias)
        residual = clean_values[..., frame, :, :].float() - prediction
        residual_rebuilt = _quantization_roundtrip(
            residual,
            bits=config.v_bits,
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
    history_length: int | None = None,
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
    reconstruction as history.  V is then fitted from the final reconstructed
    K and, when configured, iterated against its own quantized reconstruction.
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
    if (
        history_length <= 0
        or history_length > 8
        or history_length >= int(keys.shape[-3])
    ):
        raise ValueError(
            "history_length must be in [1, min(8, T-1)], got "
            f"{history_length} for T={int(keys.shape[-3])}"
        )
    if resolved.v_history_length >= int(keys.shape[-3]):
        raise ValueError(
            "config.v_history_length must be smaller than the trace frame count"
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
        reconstructed_values = _reconstruct_cross_kv_trace(
            reconstructed, values, initial_joint, resolved
        )
        for _ in range(iterations):
            fitted_cross = fit_cross_kv(
                reconstructed,
                values,
                reconstructed_values=reconstructed_values,
                history_length=resolved.v_history_length,
                history_mode="diagonal",
                ridge=cross_ridge,
                stride=resolved.stride,
                layer_ids=layer_ids,
            )
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
