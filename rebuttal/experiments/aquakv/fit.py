"""Chunked multivariate least squares, matching official ``fit_linear_regression``.

The video traces do not fit in the ``torch.stack`` that ``train_predictors.py``
uses, so the normal equations are accumulated in the same order as that
function's ``chunk_size`` path: bias column of ones, ``reg_rate * mean(|diag|)``
on every diagonal entry including the bias, then ``pinverse``.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from .official import fit_linear_regression


def fit_streaming(
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    in_features: int,
    out_features: int,
    reg_rate: float,
    out_dtype: torch.dtype,
    compute_dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(weight [out, in], bias [out])`` with ``Y ~ X @ W.T + b``."""
    if in_features < 1 or out_features < 1:
        raise ValueError("regression features must be positive")
    device = torch.device("cpu") if device is None else torch.device(device)
    cxx = torch.zeros(in_features + 1, in_features + 1, device=device, dtype=compute_dtype)
    cxy = torch.zeros(in_features + 1, out_features, device=device, dtype=compute_dtype)
    seen = 0
    for inputs, targets in batches:
        if inputs.ndim != 2 or targets.ndim != 2:
            raise ValueError("regression batches must be [rows, features]")
        if inputs.shape[1] != in_features or targets.shape[1] != out_features or inputs.shape[0] != targets.shape[0]:
            raise ValueError("regression batch shape does not match the declared features")
        if inputs.shape[0] == 0:
            continue
        xb = inputs.to(device=device, dtype=compute_dtype)
        yb = targets.to(device=device, dtype=compute_dtype)
        ones = torch.ones(xb.shape[0], 1, device=device, dtype=compute_dtype)
        xb = torch.cat([xb, ones], dim=1)
        cxx = torch.addmm(cxx, xb.T, xb, out=cxx)
        cxy = torch.addmm(cxy, xb.T, yb, out=cxy)
        seen += int(inputs.shape[0])
    if seen == 0:
        raise ValueError("regression received no rows")
    if reg_rate > 0:
        index = torch.arange(cxx.shape[0], device=device)
        cxx[index, index] += float(reg_rate) * cxx.diag().abs().mean()
    solved = (torch.pinverse(cxx) @ cxy).T.to(dtype=out_dtype)
    return solved[:, :-1].contiguous(), solved[:, -1].contiguous()


def fit_official_chunked(inputs: torch.Tensor, targets: torch.Tensor, *, reg_rate: float,
                         chunk_size: int | None, out_dtype: torch.dtype | None = None):
    """Call the vendored official fitter. Used to test the streaming accumulator."""
    weight, bias = fit_linear_regression(
        inputs, targets, reg_rate=reg_rate, fit_intercept=True,
        compute_dtype=torch.float32, chunk_size=chunk_size,
    )
    if out_dtype is not None:
        weight = weight.to(out_dtype)
        bias = bias.to(out_dtype)
    return weight, bias


def chunks(inputs: torch.Tensor, targets: torch.Tensor, chunk_size: int):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, int(inputs.shape[0]), chunk_size):
        yield inputs[start:start + chunk_size], targets[start:start + chunk_size]
