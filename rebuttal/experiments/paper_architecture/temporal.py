"""Paper-aligned temporal K ablations with decoder-visible closed-loop fitting."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from rebuttal.experiments.paper_architecture.storage import WeightTensor
from tempokv.fitting import _anchor_roundtrip, _quantization_roundtrip


@dataclass
class TemporalModel:
    kind: str
    layers: int
    heads: int
    history_length: int
    dimension: int
    bias: torch.Tensor | None = None
    weight: WeightTensor | None = None
    factor_a: WeightTensor | None = None
    factor_b: WeightTensor | None = None
    block_size: int | None = None

    def predict(self, history: list[torch.Tensor], layer: int) -> torch.Tensor:
        if not history or not 0 <= layer < self.layers:
            raise ValueError('invalid temporal history or layer')
        if history[0].ndim != 4 or history[0].shape[1] != self.heads or history[0].shape[-1] != self.dimension:
            raise ValueError('history must have shape [B,H,S,D] matching predictor')
        order = min(len(history), self.history_length)
        result = torch.zeros_like(history[0], dtype=torch.float32)
        if self.kind == 'identity':
            return history[0].float()
        w = self.weight.materialize(layer)[:, order - 1, :order] if self.weight is not None else None
        a = self.factor_a.materialize(layer)[:, order - 1, :order] if self.factor_a is not None else None
        b = self.factor_b.materialize(layer)[:, order - 1, :order] if self.factor_b is not None else None
        for lag, previous in enumerate(history[:order]):
            x = previous.float()
            if self.kind in {'paper', 'channel_refit'}:
                result += x * w[:, lag][None, :, None, :]
            elif self.kind == 'scalar':
                result += x * w[:, lag][None, :, None, :]
            elif self.kind == 'dense':
                result += torch.einsum('bhsd,hde->bhse', x, w[:, lag])
            elif self.kind == 'block':
                width = self.block_size
                blocks = self.dimension // width
                result += torch.einsum('bhsci,hcij->bhscj', x.reshape(*x.shape[:-1], blocks, width), w[:, lag]).reshape_as(result)
            elif self.kind == 'low_rank':
                hidden=torch.einsum('bhsd,hdr->bhsr',x,a[:,lag])
                result += torch.einsum('bhsr,hre->bhse',hidden,b[:,lag])
        return result if self.bias is None else result + self.bias[layer, :, order - 1][None, :, None, :]

    def storage_tensors(self) -> list[torch.Tensor]:
        result = [self.bias] if self.bias is not None else []
        for payload in (self.weight, self.factor_a, self.factor_b):
            if payload is not None:
                result.extend(payload.tensors())
        return result

    def state(self) -> dict:
        return {'kind': self.kind, 'layers': self.layers, 'heads': self.heads,
                'history_length': self.history_length, 'dimension': self.dimension,
                'block_size': self.block_size,
                'bias': None if self.bias is None else self.bias.detach().cpu(),
                **{name: None if getattr(self, name) is None else getattr(self, name).state()
                   for name in ('weight', 'factor_a', 'factor_b')}}

    def to(self, device) -> 'TemporalModel':
        return TemporalModel(self.kind, self.layers, self.heads, self.history_length, self.dimension,
                             None if self.bias is None else self.bias.to(device),
                             *(None if getattr(self, name) is None else getattr(self, name).to(device)
                               for name in ('weight', 'factor_a', 'factor_b')),
                             self.block_size)


def load_temporal(state: dict, device='cpu') -> TemporalModel:
    return TemporalModel(state['kind'], int(state['layers']), int(state['heads']),
                         int(state['history_length']), int(state['dimension']),
                         None if state['bias'] is None else state['bias'].to(device),
                         *(None if state.get(name) is None else WeightTensor.from_state(state[name], device)
                           for name in ('weight', 'factor_a', 'factor_b')),
                         state.get('block_size'))


def _replay(keys: torch.Tensor, model: TemporalModel, codec) -> torch.Tensor:
    """Recreate the frame-wise codec loop, including stored reconstruction dtype."""
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[codec.reconstruction_dtype]
    layers, batches, frames, heads, width = keys.shape
    rebuilt = torch.empty((layers, batches, frames, heads, width), device=keys.device, dtype=dtype)
    for layer in range(layers):
        rebuilt[layer, :, 0] = _anchor_roundtrip(keys[layer, :, 0], codec).to(dtype)
        for frame in range(1, frames):
            order = min(frame, model.history_length)
            history = [rebuilt[layer, :, frame-lag].float().unsqueeze(2) for lag in range(1, order+1)]
            prediction = model.predict(history, layer=layer)[:, :, 0].permute(0, 1, 2)
            target = keys[layer, :, frame].float()
            residual = target - prediction
            q = _quantization_roundtrip(residual, bits=codec.k_bits, config=codec,
                                        symmetric=codec.residual_mode == 'symmetric')
            rebuilt[layer, :, frame] = (prediction + q).to(dtype)
    return rebuilt.float()


def _solve(x: torch.Tensor, y: torch.Tensor, kind: str, dimension: int, block_size: int | None,
           ridge: float, max_gain: float):
    """Centered ridge around persistence, fitted independently for one head/order."""
    n, order, d = x.shape
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    head_penalty = ridge * xc.square().sum(0).mean().clamp_min(1e-12)
    if kind == 'scalar':
        gram = torch.einsum('nld,nmd->lm', xc, xc)
        rhs = torch.einsum('nld,nd->l', xc, yc)
        prior = torch.zeros(order, device=x.device, dtype=x.dtype); prior[0] = 1
        penalty = ridge * (gram.diag().mean().clamp_min(1e-12))
        coef = torch.linalg.solve(gram + penalty * torch.eye(order, device=x.device, dtype=x.dtype), rhs + penalty*prior)
        coef *= min(1., max_gain / max(float(coef.abs().sum()), 1e-30))
        bias = ym - (xm * coef[:, None]).sum(0)
        return coef[:, None].float(), bias.float()
    if kind == 'channel_refit':
        gram = torch.einsum('nld,nmd->dlm', xc, xc)
        rhs = torch.einsum('nld,nd->dl', xc, yc)
        penalty = head_penalty.expand(d)
        prior = torch.zeros((d, order), device=x.device, dtype=x.dtype); prior[:, 0] = 1
        coef = torch.linalg.solve(gram + penalty[:, None, None]*torch.eye(order, device=x.device, dtype=x.dtype), rhs + penalty[:, None]*prior)
        coef *= torch.clamp(max_gain / coef.abs().sum(-1).clamp_min(1e-30), max=1)[:, None]
        matrix = coef.T
        return matrix.float(), (ym-(xm*matrix).sum(0)).float()
    if kind == 'block':
        width = block_size
        blocks = d // width
        matrices = torch.zeros((order, blocks, width, width), device=x.device, dtype=x.dtype)
        for group in range(blocks):
            lo, hi = group*width, (group+1)*width
            matrices[:, group] = _dense_solve(xc[:, :, lo:hi], yc[:, lo:hi], ridge, max_gain, penalty=head_penalty)
        flat = matrices.permute(1, 0, 2, 3).reshape(blocks, order*width, width)
        mean = xm.reshape(order, blocks, width).permute(1, 0, 2).reshape(blocks, order*width)
        bias = ym - torch.einsum('bi,bij->bj', mean, flat).reshape(d)
        return matrices.float(), bias.float()
    matrices = _dense_solve(xc, yc, ridge, max_gain)
    bias = ym - torch.einsum('ld,lde->e', xm, matrices)
    return matrices.float(), bias.float()


def _dense_solve(xc, yc, ridge, max_gain, penalty=None):
    n, order, d = xc.shape
    flat = xc.reshape(n, order*d)
    gram, rhs = flat.T @ flat, flat.T @ yc
    penalty = ridge * gram.diag().mean().clamp_min(1e-12) if penalty is None else penalty
    prior = torch.zeros_like(rhs); prior[:d] = torch.eye(d, yc.shape[-1], device=xc.device, dtype=xc.dtype)
    coef = torch.linalg.solve(gram + penalty*torch.eye(order*d, device=xc.device, dtype=xc.dtype), rhs+penalty*prior)
    matrix = coef.reshape(order, d, yc.shape[-1])
    gain = matrix.abs().sum((0, 1)).clamp_min(1e-30)
    return matrix * torch.clamp(max_gain/gain, max=1)[None, None, :]


def _low_rank(x, y, rank, ridge, max_gain):
    n, order, d = x.shape
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    initial = _dense_solve(xc, yc, ridge, max_gain)
    a = torch.zeros((order, d, rank), device=x.device, dtype=x.dtype)
    b = torch.zeros((order, rank, d), device=x.device, dtype=x.dtype)
    for lag in range(order):
        u, s, vh = torch.linalg.svd(initial[lag], full_matrices=False)
        a[lag] = u[:, :rank]*s[:rank].sqrt()[None]
        b[lag] = s[:rank].sqrt()[:, None]*vh[:rank]
    # Exact rank-constrained conditional ridge per lag; repeat cyclic updates.
    penalty = ridge * (xc.square().sum()/(order*d)).clamp_min(1e-12)
    eye = torch.eye(d, device=x.device, dtype=x.dtype)
    for _ in range(3):
        for lag in range(order):
            other = sum(xc[:, j] @ a[j] @ b[j] for j in range(order) if j != lag)
            target = yc - other
            prior = eye if lag == 0 else torch.zeros_like(eye)
            gram = xc[:, lag].T@xc[:, lag]+penalty*eye
            rhs = xc[:, lag].T@target+penalty*prior
            chol=torch.linalg.cholesky(gram)
            z=torch.linalg.solve_triangular(chol,rhs,upper=False)
            u,s,vh=torch.linalg.svd(z,full_matrices=False)
            a[lag]=torch.linalg.solve_triangular(chol.T,u[:,:rank]*s[:rank].sqrt(),upper=True)
            b[lag]=s[:rank].sqrt()[:,None]*vh[:rank]
    matrix = a@b
    gain = matrix.abs().sum((0, 1)).clamp_min(1e-30)
    b *= torch.clamp(max_gain/gain, max=1)[None, None, :]
    bias = ym - torch.einsum('ld,ldr,lre->e', xm, a, b)
    return a.float(), b.float(), bias.float()


@torch.no_grad()
def build_temporal(keys, baseline, codec, spec: dict, rounds=2, ridge=.1, max_gain=1):
    if keys.ndim != 5 or not torch.isfinite(keys).all():
        raise ValueError('keys must be finite [L,N,T,H,D]')
    layers, batches, frames, heads, dimension = keys.shape
    if baseline.layers != layers or baseline.heads != heads or baseline.history_length >= frames:
        raise ValueError('baseline and calibration shape mismatch')
    if baseline.sharing != 'layer_head' or baseline.weights.shape[-1] != dimension:
        raise ValueError('paper architecture requires independent layer/head baseline')
    kind = spec['kind']; precision = spec.get('precision', 'fp32')
    if kind not in {'paper', 'identity', 'scalar', 'channel_refit', 'dense', 'block', 'low_rank'}:
        raise ValueError('unknown temporal architecture')
    if precision not in {'fp32', 'fp16', 'int8', 'int4'} or rounds < 1 or ridge < 0 or not 0 < max_gain <= 1:
        raise ValueError('invalid fitting controls or precision')
    p = baseline.history_length
    if kind == 'paper':
        model = TemporalModel(kind, layers, heads, p, dimension,
                              baseline.bias.reshape(layers, heads, p, dimension).float().clone(),
                              WeightTensor.from_tensor(baseline.weights.reshape(layers, heads, p, p, dimension), 'fp32'))
        return model, {'name': spec.get('name', kind), 'kind': kind, 'rounds': 0, 'history_source': 'production'}
    if kind == 'identity':
        return TemporalModel(kind, layers, heads, p, dimension), {'name': spec.get('name', kind), 'kind': kind, 'rounds': 0}
    block = spec.get('block_size') if kind == 'block' else None
    rank = spec.get('rank') if kind == 'low_rank' else None
    if kind == 'block' and (not isinstance(block, int) or block < 1 or dimension % block):
        raise ValueError('block size must divide head dimension')
    if kind == 'low_rank' and (not isinstance(rank, int) or not 1 <= rank <= dimension):
        raise ValueError('invalid low rank')
    shape = {'scalar': (1,), 'channel_refit': (dimension,), 'dense': (dimension, dimension),
             'block': (dimension//block, block, block) if block else (), 'low_rank': ()}[kind]
    bias = torch.zeros((layers, heads, p, dimension), device=keys.device, dtype=torch.float32)
    weights = torch.zeros((layers, heads, p, p, *shape), device=keys.device) if kind != 'low_rank' else None
    aa = torch.zeros((layers, heads, p, p, dimension, rank), device=keys.device) if kind == 'low_rank' else None
    bb = torch.zeros((layers, heads, p, p, rank, dimension), device=keys.device) if kind == 'low_rank' else None
    model = TemporalModel(kind, layers, heads, p, dimension, bias,
                          WeightTensor.from_tensor(weights, precision) if weights is not None else None,
                          WeightTensor.from_tensor(aa, precision) if aa is not None else None,
                          WeightTensor.from_tensor(bb, precision) if bb is not None else None, block)
    # First round uses quantized-anchor persistence; subsequent rounds use the
    # actual precision-specific model from the preceding fit.
    for iteration in range(rounds):
        initial = TemporalModel('identity', layers, heads, p, dimension) if iteration==0 else model
        rebuilt = _replay(keys, initial, codec)
        for layer in range(layers):
            for head in range(heads):
                for order in range(1, p+1):
                    stops = [order] if order < p else range(order, frames)
                    x = torch.cat([torch.stack([rebuilt[layer, :, frame-lag, head]
                                                 for lag in range(1, order+1)], dim=1)
                                   for frame in stops], dim=0).double()
                    y = torch.cat([keys[layer, :, frame, head] for frame in stops], dim=0).double()
                    if kind == 'low_rank':
                        a, b, intercept = _low_rank(x, y, rank, ridge, max_gain)
                        aa[layer, head, order-1, :order] = a
                        bb[layer, head, order-1, :order] = b
                    else:
                        fitted, intercept = _solve(x, y, kind, dimension, block, ridge, max_gain)
                        weights[layer, head, order-1, :order] = fitted
                    bias[layer, head, order-1] = intercept
        model = TemporalModel(kind, layers, heads, p, dimension, bias.clone(),
                              WeightTensor.from_tensor(weights, precision) if weights is not None else None,
                              WeightTensor.from_tensor(aa, precision) if aa is not None else None,
                              WeightTensor.from_tensor(bb, precision) if bb is not None else None, block)
    return model, {'name': spec.get('name', kind), 'kind': kind, 'rounds': rounds,
                   'history_source': 'quantized_reconstruction', 'ridge': ridge, 'max_gain': max_gain}
