"""AQUA-KV calibration using the upstream regression and reconstructed inputs."""
from __future__ import annotations

import torch
from torch import nn

from .quantizer import Quantizer
from .upstream.linear_utils import fit_linear_regression

SCHEMA = "aquakv.self_forcing.predictors.v1"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class AquaKVPredictor(nn.Module):
    def __init__(self, layers: int, heads: int, dim: int, *, dtype=torch.bfloat16, device="cpu"):
        super().__init__()
        if layers < 2 or min(heads, dim) < 1:
            raise ValueError("AquaKV needs at least two layers and positive head dimensions")
        self.layers, self.heads, self.dim = layers, heads, dim
        features = heads * dim
        # Initialisation must not consume the generator's random stream.
        factory = dict(device="meta", dtype=dtype)
        self.keys = nn.ModuleDict({str(i): nn.Linear(features, features, **factory) for i in range(1, layers)})
        self.values = nn.ModuleDict({str(i): nn.Linear(2*features, features, **factory) for i in range(1, layers)})
        self.to_empty(device=device)
        for parameter in self.parameters():
            nn.init.zeros_(parameter)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def predict_key(self, previous_key, layer):
        return self.keys[str(layer)](previous_key.flatten(-2).to(self.dtype)).reshape_as(previous_key)

    def predict_value(self, current_key, previous_value, layer):
        # Match upstream concatenation order exactly.
        x = torch.cat([current_key.flatten(-2), previous_value.flatten(-2)], dim=-1).to(self.dtype)
        return self.values[str(layer)](x).reshape_as(current_key)

    def parameter_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.parameters())


def validate_trace(keys, values):
    if keys.ndim != 5 or keys.shape != values.shape or keys.shape[0] < 2:
        raise ValueError("expected matching [L,records,frames,H,D] tensors with L>=2")
    if not keys.numel() or not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("empty or non-finite AquaKV calibration trace")


def measure(target, prediction, rebuilt):
    signal = target.double().square().sum().item()
    pred_error = (target.double()-prediction.double()).square().sum().item()
    error = (target.double()-rebuilt.double()).square().sum().item()
    return {"signal_sse": signal, "prediction_sse": pred_error, "error_sse": error, "numel": target.numel()}


def reconstruct_layer(k, v, layer, predictor, quantizer, previous=None, *, direct=False, metrics=True):
    pk = torch.zeros_like(k) if layer == 0 or direct else predictor.predict_key(previous[0], layer)
    kstate = quantizer.encode(k-pk, layer)
    kh = pk + quantizer.decode(kstate, layer)
    pv = torch.zeros_like(v) if layer == 0 or direct else predictor.predict_value(kh, previous[1], layer)
    vstate = quantizer.encode(v-pv, layer)
    vh = pv + quantizer.decode(vstate, layer)
    row = ({"layer_index": layer, "k": measure(k, pk, kh), "v": measure(v, pv, vh),
            "payload_bytes": kstate.storage_bytes()+vstate.storage_bytes()} if metrics else {})
    return (kh, vh), (kstate, vstate), row


@torch.no_grad()
def fit_predictors(keys, values, quantizer: Quantizer, *, parameter_dtype="bfloat16",
                   reg_rate=1e-3, fit_token_stride=1, chunk_size=4096, validation_records=0, seed=0):
    validate_trace(keys, values)
    if reg_rate < 0 or fit_token_stride < 1 or chunk_size < 1:
        raise ValueError("invalid AquaKV calibration settings")
    layers, records, _, heads, dim = keys.shape
    if not 0 <= validation_records < records:
        raise ValueError("validation_records must leave at least one training record")
    dtype = DTYPES[parameter_dtype]
    predictor = AquaKVPredictor(layers, heads, dim, dtype=dtype, device=keys.device)
    order = torch.randperm(records, generator=torch.Generator().manual_seed(seed))
    train_ids = order[validation_records:].to(keys.device)
    valid_ids = order[:validation_records].to(keys.device)
    rows, previous = [], None

    def fit_module(module, inputs, targets):
        # Upstream solver uses relative Hessian damping, including the intercept.
        x = inputs[train_ids].reshape(-1, inputs.shape[-1])[::fit_token_stride]
        y = targets[train_ids].flatten(-2).reshape(-1, heads*dim)[::fit_token_stride]
        weight, bias = fit_linear_regression(x, y, reg_rate=reg_rate, fit_intercept=True,
                                            compute_device=keys.device, compute_dtype=torch.float32,
                                            chunk_size=chunk_size)
        module.weight.copy_(weight)
        module.bias.copy_(bias)
        errors = {}
        for split, ids in (("train", train_ids), ("validation", valid_ids)):
            if len(ids):
                xx = inputs[ids].reshape(-1, inputs.shape[-1])
                yy = targets[ids].reshape(-1, heads*dim)
                numerator = denominator = 0.0
                for start in range(0, len(xx), chunk_size):
                    target = yy[start:start+chunk_size]
                    numerator += (module(xx[start:start+chunk_size]).float()-target.float()).square().sum().item()
                    denominator += target.float().square().sum().item()
                errors[split+"_prediction_nmse"] = numerator/denominator if denominator else None
        return errors

    for layer in range(layers):
        k, v = keys[layer].to(dtype), values[layer].to(dtype)
        training = {}
        if layer:
            training["k"] = fit_module(predictor.keys[str(layer)], previous[0].flatten(-2), k)
        pk = torch.zeros_like(k) if layer == 0 else predictor.predict_key(previous[0], layer)
        kh = pk + quantizer.roundtrip(k-pk, layer)
        if layer:
            x = torch.cat([kh.flatten(-2), previous[1].flatten(-2)], dim=-1)
            training["v"] = fit_module(predictor.values[str(layer)], x, v)
        previous, _, row = reconstruct_layer(k, v, layer, predictor, quantizer, previous)
        rows.append({**row, "calibration": training})
    return predictor.eval(), rows


@torch.no_grad()
def replay(keys, values, predictor, quantizer, *, direct=False):
    validate_trace(keys, values)
    if (keys.shape[0], keys.shape[-2], keys.shape[-1]) != (predictor.layers, predictor.heads, predictor.dim):
        raise ValueError("predictor does not match trace layers/heads/dim")
    rows, previous = [], None
    for layer in range(predictor.layers):
        previous, _, row = reconstruct_layer(keys[layer].to(predictor.dtype), values[layer].to(predictor.dtype),
                                             layer, predictor, quantizer, previous, direct=direct)
        rows.append(row)
    return rows


def predictor_state(predictor):
    return {"schema": SCHEMA, "layers": predictor.layers, "heads": predictor.heads, "dim": predictor.dim,
            "parameter_dtype": str(predictor.dtype).removeprefix("torch."),
            "weights": {k: v.detach().cpu() for k, v in predictor.state_dict().items()}}


def load_predictor(state, device="cpu"):
    if state.get("schema") != SCHEMA:
        raise ValueError("expected independent AquaKV predictors; TempoKV/legacy ridge files are incompatible")
    model = AquaKVPredictor(state["layers"], state["heads"], state["dim"],
                            dtype=DTYPES[state["parameter_dtype"]], device=device)
    model.load_state_dict(state["weights"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("non-finite AquaKV predictor weights")
    return model.eval()
