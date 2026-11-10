"""Real weight payloads: no persistent expanded copy for compressed weights."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


def storage_bytes(tensors):
    allocations = {}
    for tensor in tensors:
        if tensor is None or not tensor.numel():
            continue
        storage = tensor.untyped_storage()
        allocations[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(allocations.values())


@dataclass
class WeightTensor:
    precision: str
    shape: tuple
    value: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    scales: torch.Tensor | None = None

    @classmethod
    def from_tensor(cls, tensor, precision='fp32'):
        tensor = tensor.detach().float()
        if not torch.isfinite(tensor).all():
            raise ValueError('non-finite predictor weights')
        if precision in {'fp32', 'fp16'}:
            dtype = torch.float32 if precision == 'fp32' else torch.float16
            return cls(precision, tuple(tensor.shape), value=tensor.to(dtype).contiguous().clone())
        if precision not in {'int8', 'int4'} or tensor.ndim < 2:
            raise ValueError('matrix weight precision must be fp32/fp16/int8/int4')
        maximum = 127 if precision == 'int8' else 7
        amplitude = tensor.abs().amax(dim=-2, keepdim=True)
        scale = (amplitude / maximum).clamp(min=2**-24, max=65504)
        scale = torch.where(amplitude > 0, scale, torch.ones_like(scale)).to(torch.float16)
        quantized = (tensor / scale.float()).round().clamp(-maximum, maximum).to(torch.int8)
        if precision == 'int8':
            codes = quantized.contiguous()
        else:
            unsigned = quantized.to(torch.int16).bitwise_and(15).to(torch.uint8)
            if unsigned.shape[-1] % 2:
                unsigned = F.pad(unsigned, (0, 1))
            codes = (unsigned[..., ::2] | (unsigned[..., 1::2] << 4)).contiguous()
        return cls(precision, tuple(tensor.shape), codes=codes, scales=scale.contiguous())

    def materialize(self, index=None):
        if self.value is not None:
            return (self.value if index is None else self.value[index]).float()
        codes=self.codes if index is None else self.codes[index]
        scales=self.scales if index is None else self.scales[index]
        if self.precision == 'int8':
            quantized = codes.float()
        else:
            low = codes.bitwise_and(15)
            high = codes.bitwise_right_shift(4)
            unsigned = torch.stack((low, high), -1).flatten(-2)[..., :self.shape[-1]]
            quantized = unsigned.to(torch.int16)
            quantized = torch.where(quantized >= 8, quantized - 16, quantized).float()
        return quantized * scales.float()

    def tensors(self):
        return [x for x in (self.value, self.codes, self.scales) if x is not None]

    def to(self, device):
        return WeightTensor(self.precision, self.shape, **{
            name: None if getattr(self, name) is None else getattr(self, name).to(device)
            for name in ('value', 'codes', 'scales')})

    def state(self):
        return {'precision': self.precision, 'shape': list(self.shape), **{
            name: None if getattr(self, name) is None else getattr(self, name).detach().cpu()
            for name in ('value', 'codes', 'scales')}}

    @classmethod
    def from_state(cls, state, device='cpu'):
        return cls(state['precision'], tuple(state['shape']), **{
            name: None if state.get(name) is None else state[name].to(device)
            for name in ('value', 'codes', 'scales')})
