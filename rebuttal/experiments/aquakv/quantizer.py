"""Packed quantizers for the AquaKV cross-layer predictor."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch


def _cuda_kernels(device, group_size):
    if device.type != "cuda":
        return None
    try:
        from higgs_kernels import for_device
    except ImportError:
        return None
    return for_device(device, group_size)


@dataclass(frozen=True)
class HiggsConfig:
    hadamard_groupsize: int = 1024
    edenn_d: int = 2
    edenn_n: int = 16
    first_layer_edenn_d: int = 2
    first_layer_edenn_n: int = 256
    quantize_first_layer: bool = True
    chunk_size: int = 64

    def __post_init__(self):
        from .upstream.edenn import GRIDS
        group = self.hadamard_groupsize
        if group < 1 or group & (group - 1) or self.chunk_size < 1:
            raise ValueError("HIGGS requires a power-of-two Hadamard group and positive chunk_size")
        for d, n in ((self.edenn_d, self.edenn_n), (self.first_layer_edenn_d, self.first_layer_edenn_n)):
            if n > 256 or n < 2 or n & (n - 1) or n not in GRIDS.get(d, {}):
                raise ValueError("choose an upstream grid with power-of-two size <=256 (uint8 indices)")


@dataclass(frozen=True)
class TempoKVQuantizerConfig:
    k_bits: int = 4
    v_bits: int = 4
    anchor_bits: int = 4
    group_size: int = 64
    residual_mode: str = "asymmetric"
    scale_dtype: str = "bfloat16"

    def __post_init__(self):
        from tempokv.config import DTYPES, RAW_ANCHOR_BITS

        for name, maximum in (("k_bits", 8), ("v_bits", 8), ("anchor_bits", RAW_ANCHOR_BITS)):
            bits = getattr(self, name)
            if type(bits) is not int or not 1 <= bits <= maximum:
                raise ValueError(f"AquaKV TempoKV {name} must be an integer from 1 to {maximum}")
        if type(self.group_size) is not int or self.group_size < 1:
            raise ValueError("AquaKV TempoKV group_size must be a positive integer")
        if self.residual_mode not in {"symmetric", "asymmetric"}:
            raise ValueError("AquaKV TempoKV residual_mode must be symmetric or asymmetric")
        if self.scale_dtype not in DTYPES:
            raise ValueError("AquaKV TempoKV scale_dtype must be float32, float16 or bfloat16")


def quantizer_config(config):
    backend = config.get("quantizer_backend", "higgs")
    if backend == "higgs":
        if config.get("tempokv_quantizer"):
            raise ValueError("AquaKV tempokv_quantizer requires quantizer_backend=tempokv")
        return HiggsConfig(**config.get("higgs", {}))
    if backend == "tempokv":
        return TempoKVQuantizerConfig(**config.get("tempokv_quantizer", {}))
    raise ValueError("AquaKV quantizer_backend must be higgs or tempokv")


def build_quantizer(config, heads, dim, *, fitted_settings=None):
    settings = quantizer_config(config)
    if isinstance(settings, HiggsConfig):
        quantizer = Quantizer(heads * dim, settings)
    else:
        quantizer = TempoKVQuantizer(heads, dim, settings)
    if fitted_settings is not None and fitted_settings != quantizer.settings():
        raise ValueError("AquaKV quantizer settings differ from calibration; fit a matching predictor")
    return quantizer


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack each row into bytes, including six-bit centroid indices."""
    width = indices.shape[-1]
    rows = indices.reshape(-1, width).long()
    positions = torch.arange(width, device=rows.device) * bits
    byte, offset = positions // 8, positions % 8
    aligned = rows << (16 - bits - offset)
    result = torch.zeros((rows.shape[0], (width * bits + 7)//8 + 1), dtype=torch.int64, device=rows.device)
    result.scatter_add_(1, byte.expand_as(rows), aligned >> 8)
    result.scatter_add_(1, (byte + 1).expand_as(rows), aligned & 255)
    return result[:, :-1].to(torch.uint8).reshape(*indices.shape[:-1], -1)


def unpack_indices(payload: torch.Tensor, bits: int, width: int) -> torch.Tensor:
    rows = torch.nn.functional.pad(payload.reshape(-1, payload.shape[-1]).long(), (0, 1))
    positions = torch.arange(width, device=rows.device) * bits
    byte, offset = positions // 8, positions % 8
    result = (((rows[:, byte] << 8) | rows[:, byte + 1]) >> (16 - bits - offset)) & ((1 << bits) - 1)
    return result.to(torch.uint8).reshape(*payload.shape[:-1], width)


@dataclass
class HiggsState:
    indices: torch.Tensor | None
    scales: torch.Tensor
    index_shape: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype

    def tensors(self):
        return [self.scales] if self.indices is None else [self.indices, self.scales]

    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.tensors())


class Quantizer:
    """HIGGS only: no TempoKV configuration, quantization or fitting imports."""

    direct_report_key = "higgs_direct"

    def __init__(self, channels: int, config: HiggsConfig | None = None):
        from .upstream.quantizers import HiggsQuantizer

        self.channels, self.config = channels, config or HiggsConfig()
        c = self.config
        kwargs = dict(hadamard_groupsize=c.hadamard_groupsize, channel_size=channels, chunk_size=c.chunk_size)
        self.residual = HiggsQuantizer(edenn_d=c.edenn_d, edenn_n=c.edenn_n, **kwargs)
        self.first = HiggsQuantizer(edenn_d=c.first_layer_edenn_d, edenn_n=c.first_layer_edenn_n, **kwargs)

    def settings(self):
        return asdict(self.config)

    def _backend(self, layer):
        return self.first if layer == 0 else self.residual

    @torch.no_grad()
    def encode(self, value: torch.Tensor, layer: int, *, role: str = "k") -> HiggsState:
        if value.shape[-2]*value.shape[-1] != self.channels:
            raise ValueError("HIGGS channel size does not match heads * head_dim")
        if layer == 0 and not self.config.quantize_first_layer:
            return HiggsState(None, value.detach().clone(), (), tuple(value.shape), value.dtype)
        backend = self._backend(layer)
        group = backend.hadamard_groupsize
        n = self.config.first_layer_edenn_n if layer == 0 else self.config.edenn_n
        bits = int(math.log2(n))
        kernels = _cuda_kernels(value.device, group)
        if kernels is not None:
            heads, dim = value.shape[-2:]
            rows = value if value.ndim == 4 else value.reshape(1, -1, heads, dim)
            indices, scales = kernels.quantize(
                rows, backend.grid(device=value.device), backend.grid_norm(device=value.device),
                group_size=group, code_bits=bits, code_dim=backend.edenn_d, zero_code=True,
                check_finite=True,
            )
            groups, packed_width = indices.shape[-2:]
            index_width = (group + backend.edenn_d - 1) // backend.edenn_d
            indices = indices.reshape(-1, groups, packed_width)
            index_shape = (indices.shape[0], groups, index_width)
            return HiggsState(indices, scales.reshape(-1, groups), index_shape, tuple(value.shape), value.dtype)
        native = backend.quantize(value.reshape(-1, self.channels))
        if not torch.isfinite(native.scales).all():
            raise ValueError("non-finite HIGGS scales")
        indices = pack_indices(native.idx.to(value.device), bits)
        return HiggsState(indices, native.scales, tuple(native.idx.shape), tuple(value.shape), value.dtype)

    @torch.no_grad()
    def decode(self, state: HiggsState, layer: int, *, predictor: torch.Tensor | None = None):
        if state.indices is None:
            return state.scales.clone() if predictor is None else predictor + state.scales
        backend = self._backend(layer)
        group = backend.hadamard_groupsize
        n = self.config.first_layer_edenn_n if layer == 0 else self.config.edenn_n
        bits = int(math.log2(n))
        kernels = _cuda_kernels(state.scales.device, group)
        if kernels is not None and state.dtype in {torch.float16, torch.bfloat16, torch.float32} and (
            predictor is None or (
                tuple(predictor.shape) == state.shape and predictor.device == state.scales.device
                and predictor.dtype in {torch.float16, torch.bfloat16, torch.float32}
            )
        ):
            heads, dim = state.shape[-2:]
            batch, tokens = state.shape[:2] if len(state.shape) == 4 else (1, state.index_shape[0])
            output_dtype = state.dtype if predictor is None else torch.promote_types(state.dtype, predictor.dtype)
            output = torch.empty((batch, tokens, heads, dim), device=state.scales.device, dtype=output_dtype)
            kernels.dequantize_into(
                state.indices.flatten(1).unflatten(0, (batch, tokens)),
                state.scales.unflatten(0, (batch, tokens)),
                backend.grid(device=state.scales.device), output,
                group_size=group, code_bits=bits, code_dim=backend.edenn_d, half_transform=True,
                predictor=None if predictor is None else predictor.reshape(output.shape),
                residual_dtype=state.dtype,
            )
            return output.reshape(state.shape)
        from .upstream.quantizers import QuantizedTensor

        idx = unpack_indices(state.indices, bits, state.index_shape[-1])
        result = backend.dequantize(QuantizedTensor(idx, state.scales)).reshape(state.shape).to(state.dtype)
        return result if predictor is None else predictor + result

    def roundtrip(self, value, layer, *, role="k"):
        return self.decode(self.encode(value, layer, role=role), layer)

    def auxiliary_tensors(self, device):
        backends = [self.residual] + ([self.first] if self.config.quantize_first_layer else [])
        return [t for q in backends for t in (q.grid(device=device), q.grid_norm(device=device))]


@dataclass
class ScalarState:
    codes: torch.Tensor
    scale: torch.Tensor | None
    zero_point: torch.Tensor | None
    bits: int
    dtype: torch.dtype

    def tensors(self):
        return [tensor for tensor in (self.codes, self.scale, self.zero_point) if tensor is not None]

    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())


class TempoKVQuantizer:
    """TempoKV scalar payloads with AquaKV's cross-layer reconstruction."""

    direct_report_key = "tempokv_direct"

    def __init__(self, heads: int, dim: int, config: TempoKVQuantizerConfig):
        if dim % config.group_size:
            raise ValueError("AquaKV TempoKV group_size must divide head_dim")
        from tempokv import quantization
        from tempokv.config import RAW_ANCHOR_BITS

        self.heads, self.dim, self.config = heads, dim, config
        self._ops = quantization
        self._scale_dtype = getattr(torch, config.scale_dtype)
        self._raw_anchor = config.anchor_bits == RAW_ANCHOR_BITS

    def settings(self):
        return {"backend": "tempokv", **asdict(self.config)}

    @torch.no_grad()
    def encode(self, value: torch.Tensor, layer: int, *, role: str = "k") -> ScalarState:
        if value.shape[-2:] != (self.heads, self.dim):
            raise ValueError("AquaKV TempoKV input does not match configured heads and head_dim")
        config = self.config
        if layer == 0:
            bits = config.anchor_bits
            if self._raw_anchor:
                return ScalarState(value.detach().clone(), None, None, bits, value.dtype)
        else:
            bits = {"k": config.k_bits, "v": config.v_bits}[role]
        if layer > 0 and config.residual_mode == "asymmetric":
            codes, scale, zero_point = self._ops.quantize_asymmetric(
                value, bits, config.group_size, self._scale_dtype,
            )
        else:
            codes, scale = self._ops.quantize_symmetric(value, bits, config.group_size, self._scale_dtype)
            zero_point = None
        return ScalarState(codes, scale, zero_point, bits, value.dtype)

    @torch.no_grad()
    def decode(self, state: ScalarState, layer: int, *, predictor: torch.Tensor | None = None):
        if state.scale is None:
            return state.codes.clone() if predictor is None else predictor + state.codes
        args = (state.bits, self.config.group_size, self.dim, state.dtype)
        if state.zero_point is None:
            residual = self._ops.dequantize_symmetric(state.codes, state.scale, *args)
        else:
            residual = self._ops.dequantize_asymmetric(state.codes, state.scale, state.zero_point, *args)
        return residual if predictor is None else predictor + residual

    def roundtrip(self, value, layer, *, role="k"):
        return self.decode(self.encode(value, layer, role=role), layer)

    def auxiliary_tensors(self, device):
        return []
