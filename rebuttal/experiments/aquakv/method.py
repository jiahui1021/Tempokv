"""AQUA-KV sequential calibration on pre-RoPE video tokens.

One sample is one Self-Forcing video. Its tokens are the latent positions
``[frame, spatial, head]`` flattened to the channel ``num_heads * head_dim``,
which is the same axis the official linear layer reads after ``combine_heads``.
Layer 0 is quantized with the official first-layer HIGGS grid (or kept exact
when ``not_quantize_first_layer``). Later layers fit

    K_l ~ Linear(Khat_{l-1})
    V_l ~ Linear([Khat_l, Vhat_{l-1}])

on the training videos only, then every video — training and the internal
validation split — is replaced by the residual reconstruction before the next
layer. The validation videos never enter the normal equations. Evaluation
prompts are a different set and are not passed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch

from .fit import chunks, fit_streaming
from .official import HiggsHandle


DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class AquaSpec:
    num_layers: int
    num_heads: int
    head_dim: int
    hadamard_groupsize: int = 1024
    edenn_d: int = 2
    edenn_n: int = 64
    first_layer_edenn_d: int = 2
    first_layer_edenn_n: int = 256
    not_quantize_first_layer: bool = False
    prefix_tokens: int = 4
    recent_buffer_tokens: int = 128
    percdamp: float = 1e-3
    valid_nsamples: int = 32
    seed: int = 0
    activation_dtype: str = "bfloat16"
    index_packing: str = "packed"
    chunk_size: int = 4096
    higgs_chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.num_layers < 2:
            raise ValueError("AQUA-KV needs at least 2 transformer layers")
        if self.num_heads < 1 or self.head_dim < 1:
            raise ValueError("invalid attention shape")
        if self.hadamard_groupsize < 1 or self.hadamard_groupsize & (self.hadamard_groupsize - 1):
            raise ValueError("hadamard_groupsize must be a power of two")
        if self.prefix_tokens < 0 or self.recent_buffer_tokens < 0:
            raise ValueError("prefix and recent buffer must be non-negative")
        if self.valid_nsamples < 1:
            raise ValueError("valid_nsamples must be at least 1, as in train_predictors.py")
        if self.percdamp < 0:
            raise ValueError("percdamp must be non-negative")
        if self.activation_dtype not in DTYPES:
            raise ValueError("activation_dtype must be float32, float16, or bfloat16")
        if self.index_packing not in {"packed", "uint8"}:
            raise ValueError("index_packing must be packed or uint8")

    @property
    def channel(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.activation_dtype]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AquaSpec":
        known = {field: payload[field] for field in cls.__dataclass_fields__ if field in payload}
        return cls(**known)


@dataclass
class PredictorBank:
    spec: AquaSpec
    key_weight: dict[int, torch.Tensor]
    key_bias: dict[int, torch.Tensor]
    value_weight: dict[int, torch.Tensor]
    value_bias: dict[int, torch.Tensor]
    train_indices: list[int]
    valid_indices: list[int]

    def parameter_count(self) -> int:
        total = 0
        for layer in self.key_weight:
            total += self.key_weight[layer].numel() + self.key_bias[layer].numel()
            total += self.value_weight[layer].numel() + self.value_bias[layer].numel()
        return total

    def parameter_bytes(self) -> int:
        # Official ``nn.Linear`` is created in the activation dtype.
        return self.parameter_count() * torch.empty((), dtype=self.spec.dtype).element_size()


def make_quantizers(spec: AquaSpec) -> tuple[HiggsHandle, HiggsHandle | None]:
    common = dict(channel=spec.channel, hadamard_groupsize=spec.hadamard_groupsize,
                  chunk_size=spec.higgs_chunk_size, packing=spec.index_packing)
    residual = HiggsHandle(edenn_d=spec.edenn_d, edenn_n=spec.edenn_n, **common)
    first = None if spec.not_quantize_first_layer else HiggsHandle(
        edenn_d=spec.first_layer_edenn_d, edenn_n=spec.first_layer_edenn_n, **common)
    return residual, first


def flatten_tokens(tensor: torch.Tensor) -> torch.Tensor:
    """``[T, H, D]`` or ``[B, T, H, D]`` -> ``[..., H*D]`` in head-major order."""
    if tensor.shape[-2] * tensor.shape[-1] != tensor.shape[-2] * tensor.shape[-1]:
        raise ValueError("token shape must end in [head, dim]")
    return tensor.reshape(*tensor.shape[:-2], tensor.shape[-2] * tensor.shape[-1])


def unflatten_tokens(tensor: torch.Tensor, heads: int, dim: int) -> torch.Tensor:
    if tensor.shape[-1] != heads * dim:
        raise ValueError("channel does not match heads * dim")
    return tensor.reshape(*tensor.shape[:-1], heads, dim)


def apply_linear(weight: torch.Tensor, bias: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """``inputs[..., in] -> [..., out]`` in the weight dtype, then cast back."""
    flat = inputs.reshape(-1, inputs.shape[-1]).to(dtype=weight.dtype)
    out = flat @ weight.T + bias.to(dtype=weight.dtype)
    return out.to(dtype=inputs.dtype).reshape(*inputs.shape[:-1], weight.shape[0])


def split_ids(nsamples: int, valid_nsamples: int, seed: int, device: torch.device) -> tuple[list[int], list[int]]:
    if valid_nsamples >= nsamples:
        raise ValueError(
            f"valid_nsamples={valid_nsamples} leaves no training videos out of {nsamples}; "
            "official train_predictors.py uses 32 of 256"
        )
    generator = torch.Generator(device=device).manual_seed(int(seed))
    perm = torch.randperm(nsamples, generator=generator, device=device)
    train, valid = perm.split_with_sizes((nsamples - valid_nsamples, valid_nsamples))
    return train.tolist(), valid.tolist()


def _cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


@torch.no_grad()
def project_layer(
    target: torch.Tensor,
    prediction: torch.Tensor | None,
    quantizer: HiggsHandle | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return ``(reconstruction, stored indices, scales)``.

    ``prediction is None`` is layer 0 with a quantizer, or an exact copy when
    the quantizer is also None (official ``not_quantize_first_layer``).
    """
    if quantizer is None:
        if prediction is not None:
            raise ValueError("an exact layer has no residual quantizer")
        return target, None, None
    if prediction is None:
        residual = target
        pred = torch.zeros_like(target)
    else:
        residual = target - prediction
        pred = prediction
    recon_delta, stored, scales = quantizer.quantize(flatten_tokens(residual))
    recon_delta = unflatten_tokens(recon_delta, target.shape[-2], target.shape[-1])
    return pred + recon_delta, stored, scales


def _batched_rows(features: Sequence[torch.Tensor], targets: Sequence[torch.Tensor], ids: Sequence[int],
                  chunk_size: int):
    for index in ids:
        yield from chunks(features[index], targets[index], chunk_size)


@torch.no_grad()
def calibrate(samples: Sequence[tuple[torch.Tensor, torch.Tensor]], spec: AquaSpec,
              *, device: torch.device | None = None) -> tuple[PredictorBank, dict[str, Any]]:
    """Fit predictors on a list of ``(keys, values)`` with shape ``[L, T, H, D]``."""
    if len(samples) < 2:
        raise ValueError("calibration needs at least one training video and one validation video")
    device = torch.device("cpu") if device is None else torch.device(device)
    layers, tokens, heads, dim = samples[0][0].shape
    if (layers, heads, dim) != (spec.num_layers, spec.num_heads, spec.head_dim):
        raise ValueError(
            f"sample shape {(layers, heads, dim)} does not match spec "
            f"{(spec.num_layers, spec.num_heads, spec.head_dim)}"
        )
    prepared: list[tuple[torch.Tensor, torch.Tensor]] = []
    for keys, values in samples:
        if keys.shape != values.shape or keys.shape[0] != layers or keys.shape[-2:] != (heads, dim):
            raise ValueError("every calibration video must share the spec's [L, T, H, D] shape")
        if keys.shape[1] != tokens:
            raise ValueError("official AQUA-KV calibration uses one sequence length; videos differ")
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError("non-finite calibration KV")
        prepared.append((_cast(keys, spec.dtype), _cast(values, spec.dtype)))
    train_ids, valid_ids = split_ids(len(prepared), spec.valid_nsamples, spec.seed, device)
    residual, first = make_quantizers(spec)
    prev_k: list[torch.Tensor] = []
    prev_v: list[torch.Tensor] = []
    for keys, values in prepared:
        hat_k, _, _ = project_layer(keys[0], None, first)
        hat_v, _, _ = project_layer(values[0], None, first)
        prev_k.append(hat_k)
        prev_v.append(hat_v)
    bank_k_w: dict[int, torch.Tensor] = {}
    bank_k_b: dict[int, torch.Tensor] = {}
    bank_v_w: dict[int, torch.Tensor] = {}
    bank_v_b: dict[int, torch.Tensor] = {}
    report_layers = []
    channel = spec.channel
    for layer in range(1, layers):
        key_x = [flatten_tokens(prev_k[i]) for i in range(len(prepared))]
        key_y = [flatten_tokens(prepared[i][0][layer]) for i in range(len(prepared))]
        weight, bias = fit_streaming(
            _batched_rows(key_x, key_y, train_ids, spec.chunk_size),
            in_features=channel, out_features=channel, reg_rate=spec.percdamp,
            out_dtype=spec.dtype, device=device,
        )
        bank_k_w[layer] = weight
        bank_k_b[layer] = bias
        new_k: list[torch.Tensor] = []
        for index in range(len(prepared)):
            pred = unflatten_tokens(apply_linear(weight, bias, key_x[index]), heads, dim)
            hat, _, _ = project_layer(prepared[index][0][layer], pred, residual)
            new_k.append(hat)
        value_x = [torch.cat([flatten_tokens(new_k[i]), flatten_tokens(prev_v[i])], dim=-1)
                   for i in range(len(prepared))]
        value_y = [flatten_tokens(prepared[i][1][layer]) for i in range(len(prepared))]
        v_weight, v_bias = fit_streaming(
            _batched_rows(value_x, value_y, train_ids, spec.chunk_size),
            in_features=channel * 2, out_features=channel, reg_rate=spec.percdamp,
            out_dtype=spec.dtype, device=device,
        )
        bank_v_w[layer] = v_weight
        bank_v_b[layer] = v_bias
        new_v: list[torch.Tensor] = []
        for index in range(len(prepared)):
            pred = unflatten_tokens(apply_linear(v_weight, v_bias, value_x[index]), heads, dim)
            hat, _, _ = project_layer(prepared[index][1][layer], pred, residual)
            new_v.append(hat)
        report_layers.append({
            "layer": layer,
            "train_key_rows": int(sum(key_x[i].shape[0] for i in train_ids)),
            "valid_key_rows_excluded": int(sum(key_x[i].shape[0] for i in valid_ids)),
        })
        prev_k, prev_v = new_k, new_v
    bank = PredictorBank(spec, bank_k_w, bank_k_b, bank_v_w, bank_v_b, train_ids, valid_ids)
    return bank, {"layers": report_layers, "tokens_per_video": int(tokens),
                  "train_indices": train_ids, "valid_indices": valid_ids}


def state_dict(bank: PredictorBank, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from .official import OFFICIAL_COMMIT
    payload: dict[str, Any] = {
        "format": "aquakv_self_forcing_v1",
        "official_commit": OFFICIAL_COMMIT,
        "spec": bank.spec.to_dict(),
        "train_indices": bank.train_indices,
        "valid_indices": bank.valid_indices,
        "key_predictors": {str(layer): {"weight": bank.key_weight[layer].cpu(), "bias": bank.key_bias[layer].cpu()}
                           for layer in sorted(bank.key_weight)},
        "value_predictors": {str(layer): {"weight": bank.value_weight[layer].cpu(), "bias": bank.value_bias[layer].cpu()}
                             for layer in sorted(bank.value_weight)},
    }
    if extra:
        payload.update(extra)
    return payload


def bank_from_state(payload: dict[str, Any]) -> PredictorBank:
    spec = AquaSpec.from_dict(payload["spec"])
    def _load(group: dict[str, Any], name: str) -> dict[int, torch.Tensor]:
        return {int(layer): group[layer][name] for layer in group}
    keys = payload["key_predictors"]
    values = payload["value_predictors"]
    return PredictorBank(
        spec,
        _load(keys, "weight"),
        _load(keys, "bias"),
        _load(values, "weight"),
        _load(values, "bias"),
        list(payload["train_indices"]),
        list(payload["valid_indices"]),
    )


def iter_key_batches(features: Iterable[torch.Tensor], targets: Iterable[torch.Tensor], chunk_size: int):
    for feature, target in zip(features, targets, strict=True):
        yield from chunks(feature, target, chunk_size)
