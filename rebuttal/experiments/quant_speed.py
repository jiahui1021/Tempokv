"""Same50-update QAO calibration with resident data and reused forward scores.

Profiles only configurations passing quantizer/gradient checks. All51 parameter
states remain eligible for calibration-only checkpoint selection. No LS init.
"""
import argparse
from dataclasses import replace,asdict
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from tempokv.codec import _quantize,_cuda_matmul_precision
from tempokv.config import load_config
from tempokv.fitting import _reconstruct_temporal_trace,_reconstruct_cross_kv_trace,stamp_parameter_identity,fit_predictors
from rebuttal.common.trace import load_trace,load_trace_identity,check_trace_split
from rebuttal.common.manifest import write_json,file_sha256
from rebuttal.experiments.quant_objective_scratch import unfitted_pair
from rebuttal.experiments.quant_objective_gpu import Refinement,predictions
from rebuttal.experiments.quant_objective_gpu_eval_fast import evaluate


def make_roundtrip(config,compiled=False):
    assert config.residual_mode=='asymmetric' and config.scale_dtype=='bfloat16'
    qmax=(1<<config.k_bits)-1;group=config.group_size
    def dq(value):
        groups=value.float().reshape(*value.shape[:-1],-1,group)
        zero=torch.zeros_like(groups[...,:1])
        low=torch.minimum(groups.amin(-1,keepdim=True),zero)
        high=torch.maximum(groups.amax(-1,keepdim=True),zero)
        scale=(high-low).clamp_min(1e-12).div(qmax).to(torch.bfloat16).float()
        zp=torch.round(-low/scale).clamp(0,qmax).to(torch.uint8).float()
        codes=torch.round(groups/scale+zp).clamp(0,qmax).to(torch.uint8).float()
        return ((codes-zp)*scale).reshape_as(value)
    return torch.compile(dq,fullgraph=True,options={'triton.cudagraphs':False,
        'cuda.use_fast_math':False,'emulate_precision_casts':True,'force_same_precision':True}) if compiled else dq


def check_roundtrip(fn,config,device):
    g=torch.Generator(device=device).manual_seed(123)
    random=torch.randn(2,3,12,128,device=device,generator=g)
    qmax=(1<<config.k_bits)-1
    scale=torch.tensor(1/qmax,device=device).bfloat16().float()
    boundary=torch.zeros_like(random).reshape(-1,config.group_size)
    boundary[:,1]=1
    for i in range(2,config.group_size):boundary[:,i]=((i-2)%qmax+.5)*scale
    for value in (random,boundary.reshape_as(random)):
        a=value.detach().clone().requires_grad_(True);b=value.detach().clone().requires_grad_(True)
        ref=_quantize(a,bits=config.k_bits,group_size=config.group_size,mode=config.residual_mode,
                      scale_dtype=torch.bfloat16)[3]
        got=fn(b)
        assert torch.equal(ref,got),f'quantizer values changed; max error {float((ref-got).abs().max())}'
        ((a-ref).square().mean()).backward();((b-got).square().mean()).backward()
        assert torch.allclose(a.grad,b.grad,rtol=1e-5,atol=1e-7),'quantizer gradient changed'


class Fast(Refinement):
    def __init__(self,*args,dq=None,**kwargs):
        super().__init__(*args,**kwargs);self.dq=dq

    def pass_data(self,objective='quantization',*,backward=False):
        assert objective=='quantization'
        trace,c=self.trace,self.config
        layers,records,frames,heads,width=trace.shape
        denominator=layers*records*(frames-1)*heads*width*2
        k,v=self.pair();kw,kb,vw,vb,alpha=self.parameters;scalars=[]
        for layer in range(0,layers,self.layer_batch):
            stop=min(layer+self.layer_batch,layers)
            for start in range(0,records,self.batch_records):
                keys=trace.keys[layer:stop,start:start+self.batch_records]
                values=trace.values[layer:stop,start:start+self.batch_records]
                lv=replace(v,weight=v.weight[layer:stop],bias=v.bias[layer:stop],
                    history_weights=v.history_weights[layer:stop],layer_ids=trace.layer_ids[layer:stop])
                with torch.no_grad():
                    rk_hist=_reconstruct_temporal_trace(keys,k,c,residual_roundtrip=self.dq)
                    rv_hist=_reconstruct_cross_kv_trace(rk_hist,values,lv,c,residual_roundtrip=self.dq)
                with torch.set_grad_enabled(backward),_cuda_matmul_precision(self.device,'ieee'):
                    pk,pv=predictions(keys,values,rk_hist,rv_hist,kw,kb,vw,vb,alpha[layer:stop])
                    rk=keys[:,:,1:]-pk[:,:,1:];rv=values[:,:,1:]-pv[:,:,1:]
                    if self.dq is None:
                        dk=_quantize(rk,bits=c.k_bits,group_size=c.group_size,mode=c.residual_mode,scale_dtype=torch.bfloat16)[3]
                        dv=_quantize(rv,bits=c.v_bits,group_size=c.group_size,mode=c.residual_mode,scale_dtype=torch.bfloat16)[3]
                    else:dk,dv=self.dq(rk),self.dq(rv)
                    loss=((rk-dk).square().sum()+(rv-dv).square().sum())/denominator
                    if backward:loss.backward()
                    scalars.append(loss.detach())
                del keys,values,rk_hist,rv_hist,pk,pv,rk,rv,dk,dv,loss
        value=float(torch.stack(scalars).double().sum())
        return dict(objective_mse=value,residual_quant_mse=value)


def train(model,args,folder):
    optimizer=torch.optim.Adam(model.parameters,lr=.001)
    best_loss=float('inf');best=None;best_step=0;history=[]
    torch.cuda.synchronize();began=time.perf_counter()
    for step in range(args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        current=model.pass_data(backward=step<args.steps)
        loss=current['residual_quant_mse'];assert math.isfinite(loss)
        selected=loss<best_loss
        if selected:
            best_loss=loss;best_step=step;best=[p.detach().clone() for p in model.parameters]
        history.append(dict(step=step,**current,selected=selected))
        if step==args.steps:break
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters)
        lr=.001*(.1+.9*(1+math.cos(math.pi*step/max(1,args.steps-1)))/2)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();model.project()
        if step%5==0:
            print(json.dumps(dict(event='fast_update',step=step,loss=loss)),flush=True)
            write_json(folder/'progress.json',dict(history=history,elapsed_seconds=time.perf_counter()-began))
    torch.cuda.synchronize();seconds=time.perf_counter()-began
    with torch.no_grad():
        for p,value in zip(model.parameters,best):p.copy_(value)
    k,v=model.pair();k,v=stamp_parameter_identity(k,v,signature=model.config.format_signature(),coordinate_system=k.coordinate_system)
    k.save(folder/'k_temporal.npz');v.save(folder/'v_cross_kv.npz')
    write_json(folder/'codec.json',asdict(model.config))
    report=dict(status='complete',fit_seconds=seconds,updates=args.steps,checkpoint_states=args.steps+1,
        full_data_passes=args.steps+1,best_step=best_step,history=history,
        initialization='same unfitted previous-frame initialization; no LS parameters',
        gradient_approximation='unchanged codes/scale/cast gradient and detached history',
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(folder/'fit_report.json',report)
    return k,v,report


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.92,device)
    load_start=time.perf_counter();cpu=load_trace(args.calibration,device='cpu')
    load_seconds=time.perf_counter()-load_start
    split=check_trace_split(cpu,load_trace_identity(args.evaluation));assert split['prompt_identity_verified']
    setup_start=time.perf_counter();trace=replace(cpu,keys=cpu.keys.to(device),values=cpu.values.to(device));torch.cuda.synchronize()
    transfer_seconds=time.perf_counter()-setup_start
    all_results={}
    for bits in (2,4):
        folder=out/f'int{bits}';folder.mkdir(exist_ok=True)
        c=load_config(Path(args.original)/f'int{bits}/params/production/codec.json')
        k,v=unfitted_pair(cpu,c)
        prepare_start=time.perf_counter()
        reference=Refinement(trace,c,k,v,device,32,8)
        start=time.perf_counter();ref=reference.pass_data('quantization',backward=True);torch.cuda.synchronize()
        reference_seconds=time.perf_counter()-start
        grads=[p.grad.detach().cpu().clone() for p in reference.parameters]
        del reference;torch.cuda.empty_cache()
        profiles=[]
        for mode,batch in (('packed',64),('unpacked',64),('compiled',64)):
            model=None
            try:
                write_json(out/'status.json',dict(status='running',phase='profile',bits=bits,mode=mode,batch=batch))
                dq=None if mode=='packed' else make_roundtrip(c,mode=='compiled')
                if dq is not None:check_roundtrip(dq,c,device)
                model=Fast(trace,c,k,v,device,batch,30,dq=dq)
                start=time.perf_counter();probe=model.pass_data(backward=True);torch.cuda.synchronize()
                first_seconds=time.perf_counter()-start
                difference=(sum(float((p.grad.detach().cpu()-g).square().sum()) for p,g in zip(model.parameters,grads)) /
                    max(1e-30,sum(float(g.square().sum()) for g in grads)))**.5
                assert math.isclose(probe['residual_quant_mse'],ref['residual_quant_mse'],rel_tol=1e-5,abs_tol=1e-8)
                assert difference<.001,f'full gradient relative difference {difference}'
                for p in model.parameters:p.grad=None
                start=time.perf_counter();model.pass_data(backward=True);torch.cuda.synchronize()
                steady=time.perf_counter()-start
                profiles.append(dict(mode=mode,batch=batch,valid=True,first_pass_seconds=first_seconds,
                    steady_gradient_pass_seconds=steady,gradient_relative_difference=difference))
            except (RuntimeError,AssertionError) as error:
                profiles.append(dict(mode=mode,batch=batch,valid=False,error=str(error)))
            finally:
                if model is not None:del model
                gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(event='profile',bits=bits,profile=profiles[-1])),flush=True)
            write_json(folder/'profiles.json',dict(reference_pass_seconds=reference_seconds,profiles=profiles))
        candidates=[p for p in profiles if p['valid']]
        assert candidates,'No speed configuration preserved the objective/gradient'
        chosen=min(candidates,key=lambda p:p['steady_gradient_pass_seconds'])
        dq=None if chosen['mode']=='packed' else make_roundtrip(c,chosen['mode']=='compiled')
        model=Fast(trace,c,k,v,device,chosen['batch'],30,dq=dq)
        # Warm this exact callable before timed fitting; included in setup cost.
        model.pass_data(backward=True)
        for p in model.parameters:p.grad=None
        torch.cuda.synchronize();prepare_seconds=time.perf_counter()-prepare_start
        write_json(out/'status.json',dict(status='running',phase='fast_fit',bits=bits,selected=chosen))
        ko,vo,fitted=train(model,args,folder)
        del model;torch.cuda.empty_cache()
        # Remeasure original closed-form calibration on the same GPU/resident data.
        torch.cuda.synchronize();start=time.perf_counter()
        lk,lv=fit_predictors(trace.keys,trace.values,config=c,iterations=2,temporal_ridge=1.,cross_ridge=.1,
            v_history_ridge=0.,v_history_mode='diagonal',layer_ids=trace.layer_ids,
            coordinate_system=trace.metadata.get('coordinate_system','pre_rope_normalized'))
        torch.cuda.synchronize();ls_seconds=time.perf_counter()-start
        del lk,lv;torch.cuda.empty_cache()
        test=load_trace(args.evaluation,device='cpu')
        metrics=evaluate(test,c,ko,vo,device,32,8)
        original=read_previous(args.previous,bits)
        result=dict(status='complete',selected=chosen,fit=fitted,profiles=profiles,
            cpu_load_seconds=load_seconds,transfer_seconds=transfer_seconds,preparation_including_profiling_seconds=prepare_seconds,
            first_run_total_from_cpu_data_ready=transfer_seconds+prepare_seconds+fitted['fit_seconds'],
            same_gpu_ls_seconds=ls_seconds,heldout=metrics,previous_slow=original,split=split)
        write_json(folder/'result.json',result);all_results[f'int{bits}']=result
        del test,ko,vo;torch.cuda.empty_cache()
    write_json(out/'summary.json',dict(status='complete',by_bits=all_results))
    write_json(out/'status.json',dict(status='complete'))


def read_previous(root,bits):
    from rebuttal.common.manifest import read_json
    return read_json(Path(root)/f'int{bits}/comparison.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('calibration','evaluation','original','previous','output'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--steps',type=int,default=50);args=parser.parse_args()
    try:run(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
