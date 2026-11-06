"""Reference codec for architecture studies, using the shipped real quantizer.

The predictor representation is deliberately separate from production archives.
Packed states subclass PackedTensor so the frame cache can own and account for
them, but must be decoded by ``ResearchPackedTensor.decode``. No Triton layout
is claimed for these predictors. Generation must disable incremental encoding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tempokv.codec import (
    PackedKV, PackedTensor, _cuda_matmul_precision, _dequantize, _dtype, _quantize,
)
from tempokv.config import CodecConfig, config_from_mapping
from tempokv.parameters import CrossKVParameters, TemporalParameters
from tempokv.quantization import dequantize_symmetric, quantize_symmetric


SCHEMA = "tempokv.rebuttal.architecture.parameters.v1"


def group_index(sharing: str, layer: int, head: int, heads: int) -> int:
    if sharing == "shared":
        return 0
    if sharing == "layer_shared":
        return head
    if sharing == "head_shared":
        return layer
    if sharing == "layer_head":
        return layer * heads + head
    raise ValueError(f"unknown parameter sharing {sharing!r}")


def group_count(sharing: str, layers: int, heads: int) -> int:
    return {"shared": 1, "layer_shared": heads, "head_shared": layers,
            "layer_head": layers * heads}[sharing]


@dataclass
class ResearchTemporal:
    """Warm-up coefficients [group, order, lag, feature...] and vector bias."""
    kind: str
    sharing: str
    weights: torch.Tensor
    bias: torch.Tensor
    layers: int
    heads: int

    @property
    def history_length(self) -> int:
        return int(self.weights.shape[1])

    @property
    def layered(self) -> bool:
        return self.sharing in {"head_shared", "layer_head"}

    @property
    def history_weights(self) -> torch.Tensor:
        return self.weights

    @property
    def history_bias(self) -> torch.Tensor:
        return self.bias

    @classmethod
    def from_production(cls, parameters: TemporalParameters, layers: int, heads: int):
        if parameters.layered:
            w = parameters.history_weights.permute(0, 3, 1, 2, 4).reshape(
                layers * heads, parameters.history_length, parameters.history_length, -1)
            b = parameters.history_bias.permute(0, 2, 1, 3).reshape(
                layers * heads, parameters.history_length, -1)
            sharing = "layer_head"
        elif parameters.per_head:
            w = parameters.history_weights.permute(2, 0, 1, 3)
            b = parameters.history_bias.permute(1, 0, 2)
            sharing = "layer_shared"
        else:
            w, b, sharing = parameters.history_weights[None], parameters.history_bias[None], "shared"
        return cls("channel_linear", sharing, w.clone(), b.clone(), layers, heads)

    def predict(self, history: list[torch.Tensor], *, layer: int) -> torch.Tensor:
        if not history:
            raise ValueError("temporal prediction requires reconstructed history")
        if not 0 <= layer < self.layers or history[0].shape[1] != self.heads:
            raise ValueError("predictor layer/head identity does not match runtime")
        order = min(len(history), self.history_length)
        ids = [group_index(self.sharing, layer, h, self.heads) for h in range(self.heads)]
        weights = self.weights[ids, order - 1, :order].to(history[0].device)
        bias = self.bias[ids, order - 1].to(history[0].device)
        result = torch.zeros_like(history[0], dtype=torch.float32)
        for lag, previous in enumerate(history[:order]):
            if self.kind == "dense_linear":
                result += torch.matmul(previous.float(), weights[:, lag][None])
            else:
                result += previous.float() * weights[:, lag][None, :, None, :]
        return result + bias[None, :, None, :]


@dataclass
class ResearchCross:
    """Current-K map and unchanged diagonal/dense V history coefficients."""
    kind: str
    sharing: str
    bias: torch.Tensor                    # [group, D]
    history_weights: torch.Tensor         # [L,P,H,D] or [L,P,H,D,D]
    layers: int
    heads: int
    weight: torch.Tensor | None = None
    factor_a: torch.Tensor | None = None
    factor_b: torch.Tensor | None = None

    @property
    def history_length(self) -> int:
        return int(self.history_weights.shape[1])

    @property
    def history_diagonal(self) -> bool:
        return self.history_weights.ndim == 4

    @property
    def layered(self) -> bool:
        return True

    @property
    def projection_source(self) -> str:
        return "cross_kv"

    @classmethod
    def from_production(cls, parameters: CrossKVParameters, sharing: str, layers: int, heads: int):
        w = parameters.weight if parameters.layered else parameters.weight[None].expand(layers, -1, -1, -1)
        b = parameters.bias if parameters.layered else parameters.bias[None].expand(layers, -1, -1)
        hist = parameters.history_weights
        if not parameters.layered:
            hist = hist[None].expand(layers, *hist.shape)
        count = group_count(sharing, layers, heads)
        selected_w = torch.empty(count, w.shape[-2], w.shape[-1], device=w.device)
        selected_b = torch.empty(count, b.shape[-1], device=b.device)
        seen = set()
        for layer in range(layers):
            for head in range(heads):
                index = group_index(sharing, layer, head, heads)
                if index in seen:
                    if not torch.equal(selected_w[index], w[layer, head]) or not torch.equal(selected_b[index], b[layer, head]):
                        raise ValueError("production projection violates requested sharing")
                else:
                    selected_w[index], selected_b[index] = w[layer, head], b[layer, head]
                    seen.add(index)
        return cls("dense", sharing, selected_b, hist.clone(), layers, heads, selected_w)

    def project(self, source: torch.Tensor, *, layer: int, precision: str = "ieee") -> torch.Tensor:
        if not 0 <= layer < self.layers or source.shape[1] != self.heads:
            raise ValueError("predictor layer/head identity does not match runtime")
        ids = [group_index(self.sharing, layer, h, self.heads) for h in range(self.heads)]
        # Same head-major GEMM/bias ordering as the production reference codec.
        major = source.permute(1, 0, 2, 3).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        with _cuda_matmul_precision(source.device, precision):
            if self.kind == "low_rank":
                a, b = self.factor_a[ids].to(source.device), self.factor_b[ids].to(source.device)
                prediction = torch.matmul(torch.matmul(major, a[:, None]), b[:, None])
            elif self.kind == "diagonal":
                prediction = major * self.weight[ids].to(source.device)[:, None, None, :]
            else:
                prediction = torch.matmul(major, self.weight[ids].to(source.device)[:, None])
        prediction = prediction.permute(1, 0, 2, 3).contiguous()
        prediction.add_(self.bias[ids].to(source.device)[None, :, None])
        return prediction

    def add_history(self, projection: torch.Tensor, history: list[torch.Tensor], *, layer: int) -> torch.Tensor:
        result = projection
        for lag, previous in enumerate(history[:self.history_length]):
            weights = self.history_weights[layer, lag].to(previous.device)
            if self.history_diagonal:
                result = result + weights[None, :, None] * previous.float()
            else:
                result = result + torch.matmul(previous.float(), weights[None])
        return result


@dataclass(slots=True)
class ResearchPackedTensor(PackedTensor):
    """Actual codes/scales, no dense reconstruction retained in the state."""
    research_predictor: Any = None
    layer: int = 0
    activity_counts: dict[str, int] | None = None
    activity_role: str | None = None

    def decode(self, *, source: torch.Tensor | None = None,
               output_tensor: torch.Tensor | None = None, output_layout: str = "BSHD", **_: Any) -> torch.Tensor:
        dtype = _dtype(self.reconstruction_dtype)
        if output_tensor is None:
            output = torch.empty(self.shape, device=self.anchor_quant.device, dtype=dtype)
        elif output_layout == "BSHD":
            output = output_tensor.permute(0, 2, 1, 3)
        elif output_layout == "BHSD":
            output = output_tensor
        else:
            raise ValueError("output_layout must be BSHD or BHSD")
        if tuple(output.shape) != self.shape or output.dtype != dtype:
            raise ValueError("decode output buffer shape/dtype mismatch")
        anchor = self.anchor_quant.to(dtype) if self.raw_anchor else dequantize_symmetric(
            self.anchor_quant, self.anchor_scale, self.anchor_bits, self.group_size, self.shape[-1], torch.float32
        ).to(dtype)
        output[:, :, :self.unit_lengths[0]] = anchor
        history = [anchor]
        predictor = self.research_predictor
        projection = None
        if self.predictor_kind == "cross_kv":
            if source is None or tuple(source.shape) != self.shape:
                raise ValueError("cross prediction requires complete reconstructed K")
            projection = predictor.project(source, layer=self.layer, precision=self.cross_matmul_precision)
        offset, residual_offset = self.unit_lengths[0], 0
        for length in self.unit_lengths[1:]:
            current_history = [x[:, :, :length] for x in history]
            prediction = predictor.predict(current_history, layer=self.layer) if projection is None else predictor.add_history(
                projection[:, :, offset:offset + length], current_history, layer=self.layer)
            end = residual_offset + length
            residual = _dequantize(self.residual_quant[:, :, residual_offset:end],
                                  self.residual_scale[:, :, residual_offset:end],
                                  None if self.residual_zero_point is None else self.residual_zero_point[:, :, residual_offset:end],
                                  bits=self.bits, group_size=self.group_size, width=self.shape[-1], mode=self.residual_mode)
            rebuilt = (prediction + residual).to(dtype)
            output[:, :, offset:offset + length] = rebuilt
            history = [rebuilt, *history[:max(predictor.history_length - 1, 0)]]
            offset, residual_offset = offset + length, end
        if self.activity_counts is not None:
            self.activity_counts[f"decode_{self.activity_role}"] += 1
        return output


def encode_tensor(value: torch.Tensor, predictor: ResearchTemporal | ResearchCross,
                  config: CodecConfig, *, bits: int, layer: int = 0,
                  source: torch.Tensor | None = None, diagnostics: bool = False):
    """Encode BHSD with production quantization and real storage-dtype history."""
    if value.ndim != 4 or not torch.isfinite(value).all():
        raise ValueError("encoding expects finite BHSD tensor")
    if config.mirrored or config.v_predictor_mode != "cross_kv":
        raise ValueError("architecture codec supports k_to_v / cross_kv comparisons only")
    width = value.shape[-1]
    if width % config.group_size:
        raise ValueError("group_size must divide head dimension")
    stride = min(config.stride, value.shape[2])
    dtype, scale_dtype = _dtype(config.reconstruction_dtype), _dtype(config.scale_dtype)
    anchor = value[:, :, :stride]
    if config.raw_anchor:
        aq, asc = anchor.to(dtype, copy=True), anchor.new_empty(0, dtype=scale_dtype)
        rebuilt_anchor = aq
    else:
        aq, asc = quantize_symmetric(anchor, config.anchor_bits, config.group_size, scale_dtype)
        rebuilt_anchor = dequantize_symmetric(aq, asc, config.anchor_bits, config.group_size, width, torch.float32).to(dtype)
    rebuilt = torch.empty_like(value, dtype=dtype)
    rebuilt[:, :, :stride] = rebuilt_anchor
    history, lengths, codes, scales, zeros, records = [rebuilt_anchor], [stride], [], [], [], []
    projection = predictor.project(source, layer=layer, precision=config.cross_matmul_precision) if isinstance(predictor, ResearchCross) else None
    offset = stride
    while offset < value.shape[2]:
        length = min(stride, value.shape[2] - offset)
        context = [x[:, :, :length] for x in history]
        prediction = predictor.predict(context, layer=layer) if projection is None else predictor.add_history(
            projection[:, :, offset:offset + length], context, layer=layer)
        target = value[:, :, offset:offset + length].float()
        residual = target - prediction
        q, scale, zero, residual_rebuilt = _quantize(residual, bits=bits, group_size=config.group_size,
                                                  mode=config.residual_mode, scale_dtype=scale_dtype)
        current = (prediction + residual_rebuilt).to(dtype)
        rebuilt[:, :, offset:offset + length] = current
        if diagnostics:
            records.append({"offset": offset, "length": length, "prediction_sse": float(residual.double().square().sum()),
                            "quantization_sse": float((residual - residual_rebuilt).double().square().sum()),
                            "reconstruction_sse": float((target - current.float()).double().square().sum()),
                            "target_energy": float(target.double().square().sum()), "count": target.numel()})
        history = [current, *history[:max(predictor.history_length - 1, 0)]]
        lengths.append(length); codes.append(q); scales.append(scale)
        if zero is not None:
            zeros.append(zero)
        offset += length
    packed = ResearchPackedTensor(
        shape=tuple(value.shape), predictor_kind="cross_kv" if projection is not None else "temporal",
        bits=bits, anchor_bits=config.anchor_bits, group_size=config.group_size, stride=stride,
        residual_mode=config.residual_mode, reconstruction_dtype=config.reconstruction_dtype,
        unit_lengths=tuple(lengths), anchor_quant=aq, anchor_scale=asc,
        residual_quant=torch.cat(codes, 2) if codes else None,
        residual_scale=torch.cat(scales, 2) if scales else None,
        residual_zero_point=torch.cat(zeros, 2) if zeros else None,
        cross_matmul_precision=config.cross_matmul_precision, compatibility_mode=config.fit_mode,
        research_predictor=predictor, layer=layer,
    )
    return (packed, rebuilt, records) if diagnostics else (packed, rebuilt)


class ExperimentalCodec:
    """Batch-span drop-in reference encoder, with explicit custom-state decoder."""
    backend = "pytorch_research_reference"
    supports_incremental = False
    k_channel_mean = False
    primary_role, secondary_role = "k", "v"
    predicts_v_from_key = True

    def __init__(self, temporal: ResearchTemporal, cross: ResearchCross, config: CodecConfig,
                 metadata: dict[str, Any] | None = None):
        if config.mirrored or config.v_predictor_mode != "cross_kv":
            raise ValueError("architecture generation requires k_to_v cross_kv")
        self.temporal, self.v_parameters = temporal, cross
        self.config = replace(config, use_triton=False, strict_triton=False)
        self.metadata = dict(metadata or {})
        self.activity_counts = {"encode_k": 0, "encode_v": 0, "decode_k": 0, "decode_v": 0}

    @property
    def cross_kv(self):
        return self.v_parameters

    def encode_primary(self, tensor, *, layer=0, reuse_input=False):
        packed, rebuilt = encode_tensor(tensor, self.temporal, self.config, bits=self.config.k_bits, layer=layer)
        packed.activity_counts, packed.activity_role = self.activity_counts, "k"
        self.activity_counts["encode_k"] += 1
        if reuse_input:
            tensor.copy_(rebuilt); rebuilt = tensor
        return packed, rebuilt

    encode_key = encode_primary

    def encode_secondary(self, tensor, reconstructed_primary, *, layer=0, reuse_input=False):
        packed, rebuilt = encode_tensor(tensor, self.v_parameters, self.config, bits=self.config.v_bits,
                                        layer=layer, source=reconstructed_primary)
        packed.activity_counts, packed.activity_role = self.activity_counts, "v"
        self.activity_counts["encode_v"] += 1
        if reuse_input:
            tensor.copy_(rebuilt)
        return packed

    encode_value = encode_secondary

    def encode(self, key, value, *, layer=0):
        packed_k, rebuilt_k = self.encode_primary(key, layer=layer)
        return PackedKV(packed_k, self.encode_secondary(value, rebuilt_k, layer=layer))

    def decode(self, packed, *, output_dtype=None):
        k = packed.key.decode()
        return k, packed.value.decode(source=k)

    def parameter_tensors(self):
        values = [self.temporal.weights, self.temporal.bias, self.v_parameters.bias, self.v_parameters.history_weights,
                  self.v_parameters.weight, self.v_parameters.factor_a, self.v_parameters.factor_b]
        return [x for x in values if x is not None]

    def parameter_bytes(self) -> int:
        seen = {}
        for tensor in self.parameter_tensors():
            storage = tensor.untyped_storage()
            seen[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return sum(seen.values())

    def save(self, path: str | Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema": SCHEMA, "config": asdict(self.config), "layers": self.temporal.layers,
                    "heads": self.temporal.heads, "temporal_kind": self.temporal.kind,
                    "temporal_sharing": self.temporal.sharing, "cross_kind": self.v_parameters.kind,
                    "cross_sharing": self.v_parameters.sharing, "metadata": self.metadata}
        arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
                  "temporal_weights": self.temporal.weights.detach().cpu().numpy(),
                  "temporal_bias": self.temporal.bias.detach().cpu().numpy(),
                  "cross_bias": self.v_parameters.bias.detach().cpu().numpy(),
                  "cross_history": self.v_parameters.history_weights.detach().cpu().numpy()}
        for name in ("weight", "factor_a", "factor_b"):
            tensor = getattr(self.v_parameters, name)
            if tensor is not None:
                arrays[f"cross_{name}"] = tensor.detach().cpu().numpy()
        with path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu"):
        with np.load(path, allow_pickle=False) as data:
            info = json.loads(str(data["metadata"].item()))
            if info.get("schema") != SCHEMA:
                raise ValueError("unsupported research predictor format")
            tensor = lambda name: torch.from_numpy(np.array(data[name], copy=True)).to(device)
            temporal = ResearchTemporal(info["temporal_kind"], info["temporal_sharing"], tensor("temporal_weights"),
                                        tensor("temporal_bias"), info["layers"], info["heads"])
            cross = ResearchCross(info["cross_kind"], info["cross_sharing"], tensor("cross_bias"), tensor("cross_history"),
                                  info["layers"], info["heads"],
                                  **{name: tensor(f"cross_{name}") if f"cross_{name}" in data else None
                                     for name in ("weight", "factor_a", "factor_b")})
        return cls(temporal, cross, config_from_mapping(info["config"]), info.get("metadata", {}))


def load_codec_factory(path: str | Path, device: str = "cpu") -> ExperimentalCodec:
    """Callable suitable for the explicit generation codec_factory hook."""
    return ExperimentalCodec.load(path, device=device)
