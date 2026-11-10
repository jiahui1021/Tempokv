"""Paper-aligned CrossKV current-key maps with unchanged V-history terms."""
from __future__ import annotations

from dataclasses import replace

import torch

from rebuttal.adapters.experimental_codec import ResearchCross, encode_tensor
from .storage import WeightTensor


class PaperCross(ResearchCross):
    """A ResearchCross whose stored map is exactly the map used by project()."""

    def __init__(self, kind, bias, history_weights, layers, heads, precision='fp32',
                 weights=None, factors=None, block_size=None, optimization='posthoc'):
        super().__init__(kind, 'layer_head', bias.contiguous(), history_weights.contiguous(), layers, heads)
        self.precision = precision
        self.weights = weights or []
        self.factors = factors or []
        self.block_size = block_size
        self.optimization = optimization

    def project(self, source, *, layer, precision='ieee'):
        if not 0 <= layer < self.layers or source.ndim != 4 or source.shape[1] != self.heads:
            raise ValueError('source must be BHSD with matching layer/head identity')
        # Materialization is local to this call, including quantized decode timing.
        x = source.float()
        if self.kind in {'paper', 'dense_refit', 'dense_quant', 'diagonal'}:
            if self.kind == 'diagonal':
                diagonal = self.weights[0].materialize(layer)[:, 0, :]
                y = x * diagonal[None, :, None]
            else:
                w = self.weights[0].materialize(layer)
                y = torch.einsum('bhsd,hde->bhse', x, w)
        elif self.kind == 'block':
            y = torch.zeros_like(x)
            for block, start in zip(self.weights, range(0, x.shape[-1], self.block_size)):
                end = min(start + self.block_size, x.shape[-1])
                y[..., start:end] = torch.einsum('bhsd,hde->bhse',
                                                 x[..., start:end], block.materialize(layer))
        elif self.kind in {'svd', 'low_rank_refit'}:
            a, b = (part.materialize(layer) for part in self.factors)
            y = torch.einsum('bhsd,hdr->bhsr', x, a)
            y = torch.einsum('bhsr,hre->bhse', y, b)
        elif self.kind == 'shared_tucker':
            u, core, v = (part.materialize() for part in self.factors)
            y = torch.einsum('bhtd,dr->bhtr', x, u)
            y = torch.einsum('bhtr,hrq->bhtq', y, core[layer])
            y = torch.einsum('bhtq,eq->bhte', y, v)
        else:
            raise ValueError(self.kind)
        return y + self.bias[layer][None, :, None].to(y.device)

    def storage_tensors(self):
        return [self.bias, self.history_weights] + [t for w in self.weights + self.factors for t in w.tensors()]

    def to(self, device):
        return PaperCross(self.kind, self.bias.to(device), self.history_weights.to(device),
                          self.layers, self.heads, self.precision,
                          [w.to(device) for w in self.weights], [w.to(device) for w in self.factors],
                          self.block_size, self.optimization)

    def state(self):
        return {'kind': self.kind, 'precision': self.precision, 'layers': self.layers,
                'heads': self.heads, 'block_size': self.block_size, 'optimization': self.optimization,
                'bias': self.bias.detach().cpu(), 'history_weights': self.history_weights.detach().cpu(),
                'weights': [w.state() for w in self.weights], 'factors': [w.state() for w in self.factors]}


def load_cross(state, device='cpu'):
    if state.get('state_kind')=='history_cross':
        from .history import load_history_cross
        return load_history_cross(state,device)
    return PaperCross(state['kind'], state['bias'].to(device), state['history_weights'].to(device),
                      state['layers'], state['heads'], state['precision'],
                      [WeightTensor.from_state(w, device) for w in state['weights']],
                      [WeightTensor.from_state(w, device) for w in state['factors']],
                      state.get('block_size'), state.get('optimization', 'posthoc'))


def _make(kind, w, bias, baseline, precision, *, rank=None, block_size=None, optimization='posthoc'):
    layers, heads, d, _ = w.shape
    weights, factors = [], []
    if kind in {'paper', 'dense_refit', 'dense_quant', 'diagonal', 'block'}:
        if kind == 'diagonal':
            weights = [WeightTensor.from_tensor(w.diagonal(dim1=-2, dim2=-1).unsqueeze(-2), precision)]
        elif kind == 'block':
            weights = [WeightTensor.from_tensor(w[..., start:start+block_size, start:start+block_size], precision)
                       for start in range(0, d, block_size)]
        else:
            weights = [WeightTensor.from_tensor(w, precision)]
    elif kind in {'svd', 'low_rank_refit'}:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        a = u[..., :rank] * s[..., :rank].unsqueeze(-2)
        b = vh[..., :rank, :]
        factors = [WeightTensor.from_tensor(a, precision), WeightTensor.from_tensor(b, precision)]
    elif kind == 'shared_tucker':
        # Both feature-mode bases are shared across the entire layer/head table.
        u = torch.linalg.svd(w.permute(2, 0, 1, 3).reshape(d, -1), full_matrices=False)[0][:, :rank]
        v = torch.linalg.svd(w.permute(3, 0, 1, 2).reshape(d, -1), full_matrices=False)[0][:, :rank]
        core = torch.einsum('dr,lhde,es->lhrs', u, w, v)
        factors = [WeightTensor.from_tensor(u, precision), WeightTensor.from_tensor(core, precision),
                   WeightTensor.from_tensor(v, precision)]
    return PaperCross(kind, bias.clone(), baseline.history_weights.clone(), layers, heads, precision,
                      weights, factors, block_size, optimization)


def _rebuild(values, keys, model, codec):
    layers = values.shape[0]
    rebuilt = torch.empty_like(values)
    cfg = replace(codec, stride=1)
    for layer in range(layers):
        _, out = encode_tensor(values[layer].permute(0, 2, 1, 3), model, cfg, bits=cfg.v_bits,
                               layer=layer, source=keys[layer].permute(0, 2, 1, 3))
        rebuilt[layer] = out.permute(0, 2, 1, 3).float()
    return rebuilt


def _fit(keys, values, rebuilt, baseline, kind, ridge, rank, block_size):
    layers, batch, frames, heads, d = values.shape
    w = torch.zeros((layers, heads, d, d), dtype=torch.float32, device=values.device)
    bias = torch.zeros((layers, heads, d), dtype=torch.float32, device=values.device)
    for layer in range(layers):
        for head in range(heads):
            x = keys[layer, :, 1:, head].reshape(-1, d).double()
            y = values[layer, :, 1:, head].double().clone()
            for lag in range(1, min(baseline.history_length, frames - 1) + 1):
                prev = rebuilt[layer, :, :frames-lag, head].double()
                h = baseline.history_weights[layer, lag-1, head].double()
                contribution = prev * h if baseline.history_diagonal else prev @ h
                y[:, lag-1:] -= contribution[:, :frames-lag]
            y = y.reshape(-1, d)
            mx, my = x.mean(0), y.mean(0)
            xc, yc = x-mx, y-my
            gram = xc.T @ xc
            cross = xc.T @ yc
            lam = ridge * gram.diagonal().mean().clamp_min(1e-20)
            system = gram + lam * torch.eye(d, dtype=torch.float64, device=x.device)
            if kind == 'low_rank_refit':
                # Exact rank-constrained ridge in the whitened input geometry.
                chol = torch.linalg.cholesky(system)
                z = torch.linalg.solve_triangular(chol, cross, upper=False)
                u, s, vh = torch.linalg.svd(z, full_matrices=False)
                a = torch.linalg.solve_triangular(chol.T, u[:, :rank] * s[:rank], upper=True)
                fitted = a @ vh[:rank]
            elif kind == 'diagonal':
                fitted = torch.diag(cross.diagonal() / system.diagonal())
            elif kind == 'block':
                fitted = torch.zeros_like(system)
                for start in range(0, d, block_size):
                    sl = slice(start, min(start+block_size, d))
                    fitted[sl, sl] = torch.linalg.solve(system[sl, sl], cross[sl, sl])
            else:
                fitted = torch.linalg.solve(system, cross)
            w[layer, head] = fitted.float()
            bias[layer, head] = (my - mx @ fitted).float()
    return w, bias


@torch.no_grad()
def build_cross(reconstructed_keys, values, baseline: ResearchCross, codec, spec: dict,
                rounds=2, ridge=.1):
    kind = spec['kind']
    allowed = {'paper', 'dense_refit', 'dense_quant', 'svd', 'low_rank_refit',
               'diagonal', 'block', 'shared_tucker'}
    if kind not in allowed:
        raise ValueError(f'unknown CrossKV variant {kind}')
    precision = spec.get('precision', 'fp32')
    if precision not in {'fp32', 'fp16', 'int8', 'int4'}:
        raise ValueError('invalid precision')
    if baseline.sharing != 'layer_head' or baseline.weight is None:
        raise ValueError('paper baseline must have independent dense layer/head maps')
    if reconstructed_keys.shape != values.shape or values.ndim != 5:
        raise ValueError('calibration K/V must have matching [L,N,T,H,D] shape')
    layers, _, frames, heads, d = values.shape
    if frames < 2 or (layers, heads) != (baseline.layers, baseline.heads) or d != baseline.weight.shape[-1]:
        raise ValueError('calibration shape does not match paper predictor')
    rank = spec.get('rank', min(32, d))
    block_size = spec.get('block_size', min(32, d))
    if kind in {'svd', 'low_rank_refit', 'shared_tucker'} and not 1 <= rank <= d:
        raise ValueError('rank must be in [1,D]')
    if kind == 'block' and not 1 <= block_size <= d:
        raise ValueError('block_size must be in [1,D]')
    if rounds < 1 or ridge < 0:
        raise ValueError('invalid fit controls')
    w = baseline.weight.reshape(layers, heads, d, d).float()
    bias = baseline.bias.reshape(layers, heads, d).float()
    if kind == 'paper':
        if precision != 'fp32':
            raise ValueError('paper control requires fp32')
        model = _make(kind, w, bias, baseline, precision)
        return model, {'kind': kind, 'optimization': 'paper_original', 'rounds': 0}
    if kind in {'dense_quant', 'svd', 'shared_tucker'}:
        model = _make(kind, w, bias, baseline, precision, rank=rank,
                      optimization='posthoc_no_calibration_optimization')
        return model, {'kind': kind, 'optimization': model.optimization, 'rounds': 0}
    current = _make('paper', w, bias, baseline, 'fp32')
    for _ in range(rounds):
        rebuilt = _rebuild(values, reconstructed_keys, current, codec)
        w, bias = _fit(reconstructed_keys, values, rebuilt, baseline, kind, ridge, rank, block_size)
        current = _make(kind, w, bias, baseline, precision, rank=rank, block_size=block_size,
                        optimization='closed_loop_rank_constrained_ridge' if kind == 'low_rank_refit' else 'closed_loop_ridge')
    return current, {'kind': kind, 'optimization': current.optimization, 'rounds': rounds,
                     'ridge': ridge, 'rank': rank if kind == 'low_rank_refit' else None,
                     'block_size': block_size if kind == 'block' else None}
