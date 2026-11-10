"""Matched GPU refinement: prediction loss versus real residual quantization loss.

Both arms use identical Eq.5 initialization, two full-data gradient updates,
Adam settings, detached reconstructed contexts, parameter sharing, and calibration
checkpoint selection. Integer quantization codes have no gradient; stored scale
casts use PyTorch's cast derivative. This is an approximate QAO optimizer, not an
exact derivative of finite-precision storage and not the original LS solver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from tempokv.codec import _quantize, _cuda_matmul_precision
from tempokv.config import load_config
from tempokv.parameters import TemporalParameters, CrossKVParameters
from tempokv.fitting import (stamp_parameter_identity, _reconstruct_temporal_trace,
                            _reconstruct_cross_kv_trace)
from rebuttal.common.trace import load_trace, load_trace_identity, check_trace_split
from rebuttal.common.cache_metrics import load_pair, replay
from rebuttal.common.manifest import write_json, file_sha256


def predictions(keys, values, rebuilt_k, rebuilt_v, kw, kb, vw, vb, alpha):
    """Vectorized legacy arithmetic; only reconstructed contexts are detached."""
    frame_count = keys.shape[-3]
    pk = torch.zeros_like(keys)
    for order in range(1, kw.shape[0] + 1):
        stop = order + 1 if order < kw.shape[0] else frame_count
        p = torch.zeros_like(keys[:, :, order:stop])
        for lag in range(1, order + 1):
            p = p + rebuilt_k[:, :, order-lag:stop-lag] * kw[order-1, lag-1]
        pk[:, :, order:stop] = p + kb[order-1]
    # Match predict_cross_kv's batched GEMM layout and addition order.
    pv = torch.matmul(rebuilt_k.permute(0,3,1,2,4), vw[None,:,None]).permute(0,2,3,1,4)
    pv = pv + vb
    for lag in range(1, alpha.shape[1] + 1):
        pv[:, :, lag:] = pv[:, :, lag:] + rebuilt_v[:, :, :-lag] * alpha[:,lag-1,None,None]
    return pk, pv


def residual_loss(residual, config, bits, objective):
    if objective == 'prediction':
        return residual.square().sum()
    decoded = _quantize(residual, bits=bits, group_size=config.group_size,
                        mode=config.residual_mode,
                        scale_dtype=getattr(torch, config.scale_dtype))[3]
    return (residual - decoded).square().sum()


class Refinement:
    def __init__(self, trace, config, k, v, device, batch_records, layer_batch=8):
        assert config.fit_mode == 'legacy' and config.k_predictor_mode == 'shared'
        assert config.v_predictor_mode == 'cross_kv'
        assert config.cross_kv_weight_mode == 'layer_shared'
        assert config.k_history_length == 2 and config.v_history_length == 1
        # Original W and bias are tied across layers; alpha is layer specific.
        assert torch.equal(v.weight, v.weight[0:1].expand_as(v.weight))
        assert torch.equal(v.bias, v.bias[0:1].expand_as(v.bias))
        self.trace, self.config, self.device = trace, config, device
        self.k, self.v, self.batch_records = k, v, batch_records
        self.layer_batch = layer_batch
        self.parameters = [torch.nn.Parameter(t.clone().to(device)) for t in
                           (k.history_weights, k.history_bias, v.weight[0],
                            v.bias[0], v.history_weights)]

    def pair(self):
        kw, kb, vw, vb, alpha = [p.detach() for p in self.parameters]
        return replace(self.k, history_weights=kw, history_bias=kb), replace(
            self.v, weight=vw[None].expand_as(self.v.weight).contiguous(),
            bias=vb[None].expand_as(self.v.bias).contiguous(), history_weights=alpha)

    def pass_data(self, objective, *, backward=False):
        trace, config = self.trace, self.config
        layers, records, frames, heads, width = trace.shape
        denominator = layers * records * (frames-1) * heads * width * 2
        objective_sum = quant_sum = 0.0
        k, v = self.pair()
        kw, kb, vw, vb, alpha = self.parameters
        for layer in range(0,layers,self.layer_batch):
            layer_stop = min(layer+self.layer_batch,layers)
            for start in range(0, records, self.batch_records):
                keys = trace.keys[layer:layer_stop, start:start+self.batch_records].to(self.device)
                values = trace.values[layer:layer_stop, start:start+self.batch_records].to(self.device)
                lv = replace(v, weight=v.weight[layer:layer_stop], bias=v.bias[layer:layer_stop],
                             history_weights=v.history_weights[layer:layer_stop],
                             layer_ids=trace.layer_ids[layer:layer_stop])
                with torch.no_grad():
                    rebuilt_k = _reconstruct_temporal_trace(keys, k, config)
                    rebuilt_v = _reconstruct_cross_kv_trace(rebuilt_k, values, lv, config)
                with torch.set_grad_enabled(backward), _cuda_matmul_precision(self.device, 'ieee'):
                    pk, pv = predictions(keys, values, rebuilt_k, rebuilt_v,
                                         kw, kb, vw, vb, alpha[layer:layer_stop])
                    rk, rv = keys[:, :, 1:] - pk[:, :, 1:], values[:, :, 1:] - pv[:, :, 1:]
                    loss = (residual_loss(rk, config, config.k_bits, objective) +
                            residual_loss(rv, config, config.v_bits, objective)) / denominator
                    if backward:
                        loss.backward()
                    objective_sum += float(loss.detach())
                    if objective == 'quantization':
                        quant_sum += float(loss.detach())
                    else:
                        with torch.no_grad():
                            q = (residual_loss(rk, config, config.k_bits, 'quantization') +
                                 residual_loss(rv, config, config.v_bits, 'quantization')) / denominator
                            quant_sum += float(q)
                del keys, values, rebuilt_k, rebuilt_v, pk, pv, rk, rv, loss
            print(json.dumps(dict(event='layer_complete', layer=layer, layer_stop=layer_stop,
                                  objective=objective, backward=backward)), flush=True)
        return dict(objective_mse=objective_sum, residual_quant_mse=quant_sum)

    @torch.no_grad()
    def project(self):
        # Preserve original K L1 gain and diagonal V-history bounds.
        kw, _, _, _, alpha = self.parameters
        for order in range(1, kw.shape[0] + 1):
            weight = kw[order-1, :order]
            weight.mul_((1 / weight.abs().sum(0).clamp_min(1)).unsqueeze(0))
            kw[order-1, order:] = 0
        alpha.clamp_(-1, 1)


def run(args):
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This controlled comparison requires GPU')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    config = load_config(args.codec)
    trace = load_trace(args.calibration, device='cpu')
    test = load_trace_identity(args.evaluation)
    split = check_trace_split(trace, test)
    assert split['prompt_identity_verified']
    k, v = load_pair(args.k_params, args.v_params, config, 'cpu')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    controls = dict(initialization='same_original_Eq5_parameters', epochs=args.epochs,
                    optimizer='Adam', learning_rate=args.learning_rate,
                    batch_records=args.batch_records, layer_batch=args.layer_batch, updates_per_epoch=1,
                    history_source='own_reconstruction_detached', training_span_frames=trace.shape[-3],
                    checkpoint_selection='own_calibration_objective; test_unused',
                    regularization='no_added_penalty_in_either_refinement_arm',
                    constraints='K_L1_gain<=1; V_history_abs<=1',
                    parameter_sharing='shared_K; layer_shared_cross_W_bias; per_layer_V_history',
                    original_solver='closed_form_ridge; separate_reference_only',
                    matched_solver_control='prediction arm uses identical refinement optimizer',
                    quantization_gradient='integer_codes_constant; autograd_scale_and_cast_derivative',
                    device=str(device), memory_fraction=args.memory_fraction)
    write_json(output / 'controls.json', controls)
    for objective in ('prediction', 'quantization'):
        folder = output / objective
        folder.mkdir(exist_ok=True)
        model = Refinement(trace, config, k, v, device, args.batch_records,args.layer_batch)
        optimizer = torch.optim.Adam(model.parameters, lr=args.learning_rate)
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        initial = model.pass_data(objective)
        best_loss, best = initial['objective_mse'], [p.detach().clone() for p in model.parameters]
        history = [dict(epoch=0, **initial, selected=True)]
        for epoch in range(1, args.epochs+1):
            optimizer.zero_grad(set_to_none=True)
            train = model.pass_data(objective, backward=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters)
            grad_norm = sum(float(p.grad.square().sum()) for p in model.parameters) ** .5
            optimizer.step()
            model.project()
            current = model.pass_data(objective)
            selected = current['objective_mse'] < best_loss
            if selected:
                best_loss = current['objective_mse']
                best = [p.detach().clone() for p in model.parameters]
            history.append(dict(epoch=epoch, **current, gradient_norm=grad_norm, selected=selected))
            write_json(folder / 'progress.json', dict(history=history, controls=controls))
            print(json.dumps(dict(event='epoch_complete', objective=objective, **history[-1])), flush=True)
        with torch.no_grad():
            for parameter, selected in zip(model.parameters, best):
                parameter.copy_(selected)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - began
        ko, vo = model.pair()
        ko, vo = stamp_parameter_identity(ko, vo, signature=config.format_signature(),
                                         coordinate_system=k.coordinate_system)
        ko.save(folder / 'k_temporal.npz')
        vo.save(folder / 'v_cross_kv.npz')
        write_json(folder / 'codec.json', asdict(config))
        report = dict(status='complete', method='matched_GPU_full_batch_refinement',
                      objective=objective, fit_seconds=seconds, history=history, controls=controls,
                      calibration_trace=args.calibration, calibration_sha256=file_sha256(args.calibration),
                      evaluation_split=split, original_k_sha256=file_sha256(args.k_params),
                      original_v_sha256=file_sha256(args.v_params),
                      peak_memory_allocated=torch.cuda.max_memory_allocated(device))
        write_json(folder / 'fit_report.json', report)
        del model, optimizer, ko, vo, best
        torch.cuda.empty_cache()
    write_json(output / 'fit_complete.json', dict(status='complete', controls=controls))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ('calibration', 'evaluation', 'codec', 'k-params', 'v-params', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-records', type=int, default=32)
    parser.add_argument('--layer-batch', type=int, default=8)
    parser.add_argument('--memory-fraction', type=float, default=0.06)
    run(parser.parse_args())
