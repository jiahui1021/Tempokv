"""Direct GPU QAO fitting from fixed, unfitted previous-frame initialization.

No fitted parameter file is read during fitting. Original LS parameters are
loaded only after the fitted pair is frozen, for a separate held-out comparison.
The quantization-gradient approximation is the same as the prior GPU trial.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from tempokv.config import load_config
from tempokv.parameters import TemporalParameters,CrossKVParameters
from tempokv.fitting import _initial_temporal_parameters,stamp_parameter_identity
from rebuttal.common.trace import load_trace,load_trace_identity,check_trace_split
from rebuttal.common.manifest import write_json,file_sha256
from rebuttal.experiments.quant_objective_gpu import Refinement
from rebuttal.experiments.quant_objective_gpu_eval_fast import evaluate


def unfitted_pair(trace,config):
    layers,_,_,heads,width=trace.shape
    k=_initial_temporal_parameters(config.k_history_length,width,device=torch.device('cpu'),
        stride=config.stride,signature=config.format_signature())
    v=CrossKVParameters(weight=torch.zeros(layers,heads,width,width),
        bias=torch.zeros(layers,heads,width),history_weights=torch.ones(layers,1,heads,width),
        stride=config.stride,layer_ids=trace.layer_ids,codec_signature=config.format_signature())
    return stamp_parameter_identity(k,v,signature=config.format_signature(),
        coordinate_system=trace.metadata.get('coordinate_system','pre_rope_normalized'))


def fit(trace,config,folder,args,device):
    folder.mkdir(parents=True,exist_ok=True)
    k,v=unfitted_pair(trace,config)
    model=Refinement(trace,config,k,v,device,args.batch_records,args.layer_batch)
    optimizer=torch.optim.Adam(model.parameters,lr=args.learning_rate)
    controls=dict(initialization='fixed_previous_frame; K_lag1=1; cross_W=0; bias=0; V_history=1',
        fitted_parameter_files_read_during_fit=False,optimizer='Adam',max_updates=args.steps,
        learning_rate=args.learning_rate,final_learning_rate=args.learning_rate*.1,
        learning_rate_schedule='cosine',patience=args.patience,relative_tolerance=args.relative_tolerance,
        training_shape=list(trace.shape),batch_records=args.batch_records,layer_batch=args.layer_batch,
        update_scope='one full calibration gradient per update',training_span_frames=trace.shape[-3],
        checkpoint_selection='minimum calibration residual quantization MSE; test_unused',
        constraints='same K_L1<=1 and V_history_abs<=1; same parameter sharing as original',
        quantization_gradient='codes constant; autograd scale/cast derivative; reconstructed contexts detached',
        added_regularization=False,device=str(device),memory_fraction=args.memory_fraction)
    write_json(folder/'controls.json',controls)
    torch.cuda.synchronize(device);began=time.perf_counter()
    initial=model.pass_data('quantization')
    best_loss=initial['residual_quant_mse'];best=[p.detach().clone() for p in model.parameters]
    history=[dict(step=0,**initial,selected=True)]
    stalled=0;best_step=0;reason='maximum_update_budget'
    for step in range(1,args.steps+1):
        ratio=(step-1)/max(1,args.steps-1)
        lr=args.learning_rate*(.1+.9*(1+math.cos(math.pi*ratio))/2)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        model.pass_data('quantization',backward=True)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters)
        grad_norm=sum(float(p.grad.square().sum()) for p in model.parameters)**.5
        optimizer.step();model.project()
        current=model.pass_data('quantization');loss=current['residual_quant_mse']
        assert math.isfinite(loss)
        improvement=best_loss-loss
        significant=improvement>abs(best_loss)*args.relative_tolerance
        selected=loss<best_loss
        if selected:best_loss=loss;best_step=step;best=[p.detach().clone() for p in model.parameters]
        stalled=0 if significant else stalled+1
        history.append(dict(step=step,**current,gradient_norm=grad_norm,learning_rate=lr,selected=selected))
        torch.cuda.synchronize(device)
        write_json(folder/'progress.json',dict(status='running',controls=controls,history=history,
            best_step=best_step,elapsed_seconds=time.perf_counter()-began))
        print(json.dumps(dict(event='scratch_update',bits=config.k_bits,**history[-1])),flush=True)
        if stalled>=args.patience:reason='calibration_improvement_below_tolerance';break
    with torch.no_grad():
        for parameter,value in zip(model.parameters,best):parameter.copy_(value)
    torch.cuda.synchronize(device);seconds=time.perf_counter()-began
    ko,vo=model.pair()
    ko,vo=stamp_parameter_identity(ko,vo,signature=config.format_signature(),coordinate_system=k.coordinate_system)
    ko.save(folder/'k_temporal.npz');vo.save(folder/'v_cross_kv.npz')
    write_json(folder/'codec.json',asdict(config))
    result=dict(status='complete',method='direct_QAO_from_unfitted_initialization',controls=controls,
        fit_seconds=seconds,best_step=best_step,updates_run=len(history)-1,stop_reason=reason,
        initial_calibration_mse=initial['residual_quant_mse'],best_calibration_mse=best_loss,history=history,
        k_sha256=file_sha256(folder/'k_temporal.npz'),v_sha256=file_sha256(folder/'v_cross_kv.npz'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    write_json(folder/'fit_report.json',result)
    del model,optimizer,best;torch.cuda.empty_cache()
    return ko,vo,result


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    trace=load_trace(args.calibration,device='cpu')
    split=check_trace_split(trace,load_trace_identity(args.evaluation));assert split['prompt_identity_verified']
    calibration_hash=file_sha256(trace.path)
    reports={}
    for bits in (2,4):
        config=load_config(Path(args.original)/f'int{bits}/params/production/codec.json')
        assert config.k_bits==config.v_bits==bits
        write_json(out/'status.json',dict(status='running',phase='from_scratch_fit',bits=bits))
        ko,vo,fit_report=fit(trace,config,out/f'int{bits}',args,device)
        fit_report.update(calibration_sha256=calibration_hash,evaluation_split=split)
        write_json(out/f'int{bits}/fit_report.json',fit_report)
        # All training decisions and parameter hashes are frozen before loading test values.
        write_json(out/'status.json',dict(status='running',phase='frozen_parameter_evaluation',bits=bits))
        test=load_trace(args.evaluation,device='cpu')
        scratch=evaluate(test,config,ko,vo,device,args.batch_records,args.layer_batch)
        from rebuttal.common.cache_metrics import load_pair
        original=Path(args.original)/f'int{bits}/params/production'
        oldk,oldv=load_pair(original/'k_temporal.npz',original/'v_cross_kv.npz',config,device)
        baseline=evaluate(test,config,oldk,oldv,device,args.batch_records,args.layer_batch)
        result=dict(status='complete',bits=bits,scope='non_anchor_including_warmup; span24; matched_GPU_batches',
            split=split,test_sha256=file_sha256(test.path),original_LS=baseline,scratch_QAO=scratch,fit=fit_report)
        write_json(out/f'int{bits}/comparison.json',result);reports[f'int{bits}']=result
        del test,ko,vo,oldk,oldv;torch.cuda.empty_cache()
    write_json(out/'summary.json',dict(status='complete',by_bits=reports))
    write_json(out/'status.json',dict(status='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('calibration','evaluation','original','output'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--steps',type=int,default=50);parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--relative-tolerance',type=float,default=1e-4)
    parser.add_argument('--learning-rate',type=float,default=1e-3)
    parser.add_argument('--batch-records',type=int,default=32);parser.add_argument('--layer-batch',type=int,default=8)
    parser.add_argument('--memory-fraction',type=float,default=.06)
    args=parser.parse_args()
    try:run(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
