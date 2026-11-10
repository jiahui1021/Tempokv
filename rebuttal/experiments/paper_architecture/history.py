"""V-history ablations that retain the paper's current-K CrossKV map."""
from __future__ import annotations

from dataclasses import replace

import torch

from rebuttal.adapters.experimental_codec import encode_tensor
from .cross import PaperCross
from .storage import WeightTensor


class HistoryCross(PaperCross):
    def __init__(self, bias, layers, heads, cross_weight, history_kind, history_parts,
                 precision='fp32', block_size=None, optimization='paper_original'):
        # The paper VH1 tensor is deliberately absent from this model's state.
        empty = torch.empty(0, device=bias.device)
        super().__init__('paper', bias, empty, layers, heads, 'fp32',
                         weights=[cross_weight], optimization=optimization)
        self.history_kind = history_kind
        self.history_parts = history_parts
        self.history_precision = precision
        self.block_size = block_size

    @property
    def history_length(self):
        return 1

    def add_history(self, projection, history, *, layer):
        if not history:
            return projection
        x = history[0].float()
        if self.history_kind == 'paper':
            y = x * self.history_parts[0].materialize(layer)[None, :, None]
        elif self.history_kind == 'scalar':
            y = x * self.history_parts[0].materialize(layer)[:, 0, 0][None, :, None, None]
        elif self.history_kind == 'dense':
            y = torch.einsum('bhsd,hde->bhse', x, self.history_parts[0].materialize(layer))
        elif self.history_kind == 'block':
            y = torch.zeros_like(x)
            for part, start in zip(self.history_parts, range(0, x.shape[-1], self.block_size)):
                end = min(start+self.block_size, x.shape[-1])
                y[..., start:end] = torch.einsum('bhsd,hde->bhse',
                                                 x[..., start:end], part.materialize(layer))
        elif self.history_kind == 'low_rank':
            a, b = (part.materialize(layer) for part in self.history_parts)
            y = torch.einsum('bhsd,hdr->bhsr', x, a)
            y = torch.einsum('bhsr,hre->bhse', y, b)
        else:
            raise ValueError(self.history_kind)
        return projection + y

    def storage_tensors(self):
        return [self.bias] + [t for part in self.weights + self.history_parts for t in part.tensors()]

    def state(self):
        return {'state_kind': 'history_cross', 'kind': 'paper', 'layers': self.layers,
                'heads': self.heads, 'bias': self.bias.detach().cpu(),
                'cross_weight': self.weights[0].state(),
                'history_model': {'kind': self.history_kind, 'precision': self.history_precision,
                                  'block_size': self.block_size,
                                  'parts': [part.state() for part in self.history_parts]},
                'optimization': self.optimization}

    def to(self, device):
        return HistoryCross(self.bias.to(device), self.layers, self.heads, self.weights[0].to(device),
                            self.history_kind, [part.to(device) for part in self.history_parts],
                            self.history_precision, self.block_size, self.optimization)


def load_history_cross(state, device='cpu'):
    history = state['history_model']
    return HistoryCross(state['bias'].to(device), state['layers'], state['heads'],
                        WeightTensor.from_state(state['cross_weight'], device), history['kind'],
                        [WeightTensor.from_state(part, device) for part in history['parts']],
                        history['precision'], history.get('block_size'), state.get('optimization', 'paper_original'))


def _make(baseline, kind, precision, bias, weight, block_size=None, rank=1, optimization='paper_original'):
    layers, heads, d = bias.shape
    if kind == 'paper':
        parts = [WeightTensor.from_tensor(weight, 'fp32')]
    elif kind == 'scalar':
        parts = [WeightTensor.from_tensor(weight, precision)]
    elif kind == 'dense':
        parts = [WeightTensor.from_tensor(weight, precision)]
    elif kind == 'block':
        parts = [WeightTensor.from_tensor(weight[..., start:start+block_size, start:start+block_size], precision)
                 for start in range(0, d, block_size)]
    elif kind == 'low_rank':
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        parts = [WeightTensor.from_tensor(u[..., :rank] * s[..., :rank].unsqueeze(-2), precision),
                 WeightTensor.from_tensor(vh[..., :rank, :], precision)]
    else:
        raise ValueError(kind)
    return HistoryCross(bias, layers, heads, baseline.weights[0].to(bias.device), kind, parts,
                        precision, block_size, optimization)


def _rebuild(keys, values, current, codec):
    cfg = replace(codec, stride=1)
    rebuilt = torch.empty_like(values)
    for layer in range(values.shape[0]):
        _, out = encode_tensor(values[layer].permute(0, 2, 1, 3), current, cfg, bits=cfg.v_bits,
                               layer=layer, source=keys[layer].permute(0, 2, 1, 3))
        rebuilt[layer] = out.permute(0, 2, 1, 3).float()
    return rebuilt


def _fit(keys, values, rebuilt, baseline, kind, ridge, block_size, rank):
    layers, _, frames, heads, d = values.shape
    weight_shape = (layers, heads, 1, 1) if kind == 'scalar' else (layers, heads, d, d)
    weight = torch.zeros(weight_shape, dtype=torch.float32, device=values.device)
    bias = torch.zeros((layers, heads, d), dtype=torch.float32, device=values.device)
    for layer in range(layers):
        # Keep the paper's current-K map and its original beta fixed in the fit target.
        projected = baseline.project(keys[layer].permute(0, 2, 1, 3), layer=layer).permute(0, 2, 1, 3)
        for head in range(heads):
            x = rebuilt[layer, :, :-1, head].reshape(-1, d).double()
            y = (values[layer, :, 1:, head] - projected[:, 1:, head]).reshape(-1, d).double()
            mx, my = x.mean(0), y.mean(0)
            xc, yc = x-mx, y-my
            gram, cross = xc.T @ xc, xc.T @ yc
            lam = ridge * gram.diagonal().mean().clamp_min(1e-20)
            system = gram + lam * torch.eye(d, dtype=torch.float64, device=x.device)
            if kind == 'scalar':
                # A scalar represents D tied diagonal coefficients, hence D penalties.
                value = cross.diagonal().sum() / (gram.diagonal().sum() + lam*d)
                fitted = torch.eye(d, dtype=torch.float64, device=x.device) * value
                weight[layer, head, 0, 0] = value.float()
            elif kind == 'block':
                fitted = torch.zeros_like(system)
                for start in range(0, d, block_size):
                    sl = slice(start, min(start+block_size, d))
                    fitted[sl, sl] = torch.linalg.solve(system[sl, sl], cross[sl, sl])
                weight[layer, head] = fitted.float()
            elif kind == 'low_rank':
                chol = torch.linalg.cholesky(system)
                z = torch.linalg.solve_triangular(chol, cross, upper=False)
                u, s, vh = torch.linalg.svd(z, full_matrices=False)
                a = torch.linalg.solve_triangular(chol.T, u[:, :rank] * s[:rank], upper=True)
                fitted = a @ vh[:rank]
                weight[layer, head] = fitted.float()
            else:
                fitted = torch.linalg.solve(system, cross)
                weight[layer, head] = fitted.float()
            bias[layer, head] = (baseline.bias[layer, head].double() + my - mx @ fitted).float()
    return weight, bias


@torch.no_grad()
def build_history(keys, values, baseline: PaperCross, codec, spec: dict, rounds=2, ridge=.1):
    kind = spec['kind']
    if kind not in {'paper', 'scalar', 'dense', 'block', 'low_rank'}:
        raise ValueError(f'unknown V-history variant {kind}')
    precision = spec.get('precision', 'fp32')
    if precision not in {'fp32', 'fp16'}:
        raise ValueError('V-history precision must be fp32 or fp16')
    if not isinstance(baseline, PaperCross) or baseline.kind != 'paper' or baseline.history_length != 1:
        raise ValueError('baseline must be paper CrossKV with VH1')
    if keys.shape != values.shape or keys.ndim != 5:
        raise ValueError('calibration K/V must match [L,N,T,H,D]')
    layers, _, frames, heads, d = values.shape
    if frames < 2 or (layers, heads, d) != (baseline.layers, baseline.heads, baseline.bias.shape[-1]):
        raise ValueError('calibration shape does not match paper predictor')
    block_size = int(spec.get('block_size', 2))
    rank = int(spec.get('rank', 1))
    if kind == 'block' and not 1 <= block_size <= d:
        raise ValueError('invalid block_size')
    if kind == 'low_rank' and not 1 <= rank <= d:
        raise ValueError('invalid rank')
    if rounds < 1 or ridge < 0:
        raise ValueError('invalid fit controls')
    if kind == 'paper':
        if precision != 'fp32':
            raise ValueError('paper history requires fp32')
        original = baseline.history_weights[:, 0].float()
        model = _make(baseline, kind, precision, baseline.bias.clone(), original)
        return model, {'kind': kind, 'optimization': 'paper_original', 'rounds': 0}
    current = _make(baseline, 'paper', 'fp32', baseline.bias.clone(), baseline.history_weights[:, 0].float())
    for _ in range(rounds):
        rebuilt = _rebuild(keys, values, current, codec)
        weight, bias = _fit(keys, values, rebuilt, baseline, kind, ridge, block_size, rank)
        current = _make(baseline, kind, precision, bias, weight, block_size, rank,
                        'closed_loop_rank_constrained_ridge' if kind == 'low_rank' else 'closed_loop_ridge')
    return current, {'kind': kind, 'optimization': current.optimization, 'rounds': rounds,
                     'ridge': ridge, 'block_size': block_size if kind == 'block' else None,
                     'rank': rank if kind == 'low_rank' else None}
