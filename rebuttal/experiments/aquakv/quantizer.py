"""Independent packed HIGGS wrapper around the pinned AQUA-KV implementation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch

from .upstream.quantizers import HiggsQuantizer, QuantizedTensor


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
        native = self._backend(layer).quantize(value.reshape(-1, self.channels))
        if not torch.isfinite(native.scales).all():
            raise ValueError("non-finite HIGGS scales")
        n = self.config.first_layer_edenn_n if layer == 0 else self.config.edenn_n
        indices = pack_indices(native.idx.to(value.device), int(math.log2(n)))
        return HiggsState(indices, native.scales, tuple(native.idx.shape), tuple(value.shape), value.dtype)

    @torch.no_grad()
    def decode(self, state: HiggsState, layer: int):
        if state.indices is None:
            return state.scales.clone()
        n = self.config.first_layer_edenn_n if layer == 0 else self.config.edenn_n
        idx = unpack_indices(state.indices, int(math.log2(n)), state.index_shape[-1])
        result = self._backend(layer).dequantize(QuantizedTensor(idx, state.scales))
        return result.reshape(state.shape).to(state.dtype)

    def roundtrip(self, value, layer):
        return self.decode(self.encode(value, layer), layer)

    def auxiliary_tensors(self, device):
        backends = [self.residual] + ([self.first] if self.config.quantize_first_layer else [])
        return [t for q in backends for t in (q.grid(device=device), q.grid_norm(device=device))]
