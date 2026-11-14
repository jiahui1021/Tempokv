"""Independent packed HIGGS wrapper around the pinned AQUA-KV implementation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch

from .upstream.quantizers import HiggsQuantizer, QuantizedTensor


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
    def __init__(self, channels: int, config: HiggsConfig | None = None):
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
    def encode(self, value: torch.Tensor, layer: int) -> HiggsState:
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
        idx = unpack_indices(state.indices, bits, state.index_shape[-1])
        result = backend.dequantize(QuantizedTensor(idx, state.scales)).reshape(state.shape).to(state.dtype)
        return result if predictor is None else predictor + result

    def roundtrip(self, value, layer):
        return self.decode(self.encode(value, layer), layer)

    def auxiliary_tensors(self, device):
        backends = [self.residual] + ([self.first] if self.config.quantize_first_layer else [])
        return [t for q in backends for t in (q.grid(device=device), q.grid_norm(device=device))]
