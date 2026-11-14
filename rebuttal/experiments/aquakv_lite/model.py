"""AQUA-KV (arXiv 2501.19392) cross-layer predictor, adapted to video K/V traces.

Per layer l (consecutive trace layers):
  layer 0:  K̂_0 = Q(K_0),                     V̂_0 = Q(V_0)
  l >= 1:   K̃_l = [K̂_{l-1}, 1] A_l,           K̂_l = K̃_l + Q(K_l - K̃_l)
            Ṽ_l = [V̂_{l-1}, K̂_l, 1] B_l,      V̂_l = Ṽ_l + Q(V_l - Ṽ_l)
Features are the full per-token hidden vector (all H heads x D channels), so the
predictor mixes heads like the paper's per-layer linear regressor.  Tokens are the
video latent positions [records, frames]; frame/head layout is preserved on output.
A_l, B_l are ridge least squares fitted sequentially on calibration data using the
*reconstructed* (quantized) inputs, i.e. the same recursion used at encode time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from tempokv.quantization import roundtrip

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass(frozen=True)
class Quantizer:
    """Residual quantizer; mirrors TempoKV's residual path (tempokv.quantization)."""
    bits: int
    group_size: int = 64
    mode: str = "asymmetric"
    scale_dtype: str = "bfloat16"
    reconstruction_dtype: str = "bfloat16"

    def __post_init__(self):
        if isinstance(self.bits, bool) or not 1 <= int(self.bits) <= 8:
            raise ValueError("bits must be in [1,8]")
        if self.mode not in {"symmetric", "asymmetric"}:
            raise ValueError("mode must be symmetric or asymmetric")
        if self.scale_dtype not in DTYPES or self.reconstruction_dtype not in DTYPES:
            raise ValueError("unknown dtype")

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """value [..., H, D] -> dequantized float32 of same shape."""
        if value.shape[-1] % self.group_size:
            raise ValueError("group_size must divide head_dim")
        return roundtrip(value, bits=self.bits, group_size=self.group_size,
                         scale_dtype=DTYPES[self.scale_dtype], symmetric=self.mode == "symmetric")

    def store(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(DTYPES[self.reconstruction_dtype]).float()

    def bytes_per_element(self) -> float:
        meta = DTYPE_BYTES[self.scale_dtype] * 8 + (self.bits if self.mode == "asymmetric" else 0)
        return (self.bits + meta / self.group_size) / 8


def _check(keys: torch.Tensor, values: torch.Tensor) -> None:
    if keys.ndim != 5 or keys.shape != values.shape:
        raise ValueError("keys/values must be matching [L,records,frames,H,D] tensors")
    if keys.shape[0] < 2:
        raise ValueError("AQUA-KV cross-layer prediction needs at least 2 layers")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("non-finite trace values")


def _flat(x: torch.Tensor) -> torch.Tensor:  # [R,T,H,D] -> [N, H*D]
    return x.reshape(-1, x.shape[-2] * x.shape[-1]).double()


def ridge(inputs: torch.Tensor, target: torch.Tensor, ridge: float, bias: bool = True) -> torch.Tensor:
    """Solve min ||[X,1]W - Y||^2 + ridge*||W_x||^2 (bias unregularised). Returns W [F(+1), Fo]."""
    x = inputs.double()
    if bias:
        x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], 1)
    gram = x.T @ x
    reg = torch.full((x.shape[1],), float(ridge), dtype=x.dtype, device=x.device)
    if bias:
        reg[-1] = 0.0
    gram = gram + torch.diag(reg)
    return torch.linalg.lstsq(gram, x.T @ target.double()).solution


def apply(weight: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    w = weight.to(inputs.dtype)
    bias = w.shape[0] == inputs.shape[1] + 1
    out = inputs @ (w[:-1] if bias else w)
    return out + w[-1] if bias else out


@dataclass
class AquaKVPredictor:
    key_weights: list[torch.Tensor] = field(default_factory=list)    # layer l-1 -> [F+1, F]
    value_weights: list[torch.Tensor] = field(default_factory=list)  # [2F+1, F]
    heads: int = 0
    dim: int = 0
    ridge: float = 1e-2
    bias: bool = True

    def parameter_count(self) -> int:
        return sum(w.numel() for w in self.key_weights + self.value_weights)

    def parameter_bytes(self, dtype: str = "bfloat16") -> int:
        return self.parameter_count() * DTYPE_BYTES[dtype]

    def state_dict(self) -> dict[str, Any]:
        return {"key_weights": [w.float().cpu() for w in self.key_weights],
                "value_weights": [w.float().cpu() for w in self.value_weights],
                "heads": self.heads, "dim": self.dim, "ridge": self.ridge, "bias": self.bias,
                "method": "aquakv_lite_group_quant_ridge"}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "AquaKVPredictor":
        return cls(list(state["key_weights"]), list(state["value_weights"]), int(state["heads"]),
                   int(state["dim"]), float(state["ridge"]), bool(state["bias"]))


def _metrics_init() -> dict[str, float]:
    return {"signal_sse": 0.0, "prediction_sse": 0.0, "quant_error_sse": 0.0, "error_sse": 0.0, "numel": 0}


def _accumulate(m, target, pred, residual_q, rebuilt):
    residual = target - pred
    m["signal_sse"] += float(target.double().square().sum())
    m["prediction_sse"] += float(residual.double().square().sum())
    m["quant_error_sse"] += float((residual - residual_q).double().square().sum())
    m["error_sse"] += float((target - rebuilt).double().square().sum())
    m["numel"] += target.numel()


@torch.no_grad()
def encode(keys: torch.Tensor, values: torch.Tensor, quantizer: Quantizer, *,
           predictor: AquaKVPredictor | None = None, fit: bool = False,
           ridge_value: float = 1e-2, bias: bool = True, fit_token_stride: int = 1,
           teacher_forcing: bool = False) -> tuple[AquaKVPredictor, dict[str, Any], torch.Tensor, torch.Tensor]:
    """Run the AQUA recursion over all layers.

    fit=True fits each layer's predictor on the reconstructed inputs just produced
    (sequential, as in the paper), then encodes that layer with it.  fit=False
    requires a predictor and never touches its weights (heldout evaluation).
    teacher_forcing=True feeds clean K_{l-1}/V_{l-1}/K_l instead (diagnostic only).
    Returns (predictor, per-layer metrics, K̂, V̂) with K̂/V̂ shaped like the input.
    """
    _check(keys, values)
    layers, records, frames, heads, dim = keys.shape
    if fit == (predictor is not None):
        raise ValueError("pass exactly one of fit=True or a fitted predictor")
    if predictor is None:
        predictor = AquaKVPredictor(heads=heads, dim=dim, ridge=ridge_value, bias=bias)
    elif (predictor.heads, predictor.dim) != (heads, dim) or len(predictor.key_weights) != layers - 1:
        raise ValueError(f"predictor shape (L={len(predictor.key_weights)+1},H={predictor.heads},D={predictor.dim}) "
                         f"does not match trace {tuple(keys.shape)}")
    q = quantizer
    kh = torch.empty_like(keys, dtype=torch.float32)
    vh = torch.empty_like(values, dtype=torch.float32)
    rows = []
    for layer in range(layers):
        mk, mv = _metrics_init(), _metrics_init()
        k, v = keys[layer].float(), values[layer].float()
        if layer == 0:
            pk = torch.zeros_like(k)
            pv = torch.zeros_like(v)
        else:
            src_k = keys[layer - 1].float() if teacher_forcing else kh[layer - 1]
            xk = _flat(src_k)
            if fit:
                sel = slice(None, None, fit_token_stride)
                predictor.key_weights.append(ridge(xk[sel], _flat(k)[sel], ridge_value, bias))
            pk = apply(predictor.key_weights[layer - 1], xk).float().reshape(k.shape)
        rk = q(k - pk)
        kh[layer] = q.store(pk + rk)
        _accumulate(mk, k, pk, rk, kh[layer])
        if layer:
            src_v = values[layer - 1].float() if teacher_forcing else vh[layer - 1]
            cur_k = k if teacher_forcing else kh[layer]
            xv = torch.cat([_flat(src_v), _flat(cur_k)], 1)
            if fit:
                predictor.value_weights.append(ridge(xv[sel], _flat(v)[sel], ridge_value, bias))
            pv = apply(predictor.value_weights[layer - 1], xv).float().reshape(v.shape)
        rv = q(v - pv)
        vh[layer] = q.store(pv + rv)
        _accumulate(mv, v, pv, rv, vh[layer])
        rows.append({"layer_index": layer, "k": mk, "v": mv})
    return predictor, rows, kh, vh


@torch.no_grad()
def direct(keys: torch.Tensor, values: torch.Tensor, quantizer: Quantizer):
    """Raw-residual baseline: no prediction, quantize K/V directly with the same quantizer."""
    _check(keys, values)
    rows = []
    for layer in range(keys.shape[0]):
        row = {"layer_index": layer}
        for name, x in (("k", keys[layer].float()), ("v", values[layer].float())):
            m = _metrics_init()
            r = quantizer(x)
            _accumulate(m, x, torch.zeros_like(x), r, quantizer.store(r))
            row[name] = m
        rows.append(row)
    return rows
