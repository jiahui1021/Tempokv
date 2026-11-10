"""Two-GPU full-gradient QAO: same data,50 updates and51 scored states."""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from tempokv.config import load_config
from tempokv.fitting import stamp_parameter_identity
from rebuttal.common.trace import load_trace,load_trace_identity,check_trace_split
from rebuttal.common.manifest import read_json,write_json
from rebuttal.experiments.quant_objective_scratch import unfitted_pair
from rebuttal.experiments.quant_speed import Fast,make_roundtrip
from rebuttal.experiments.quant_objective_gpu_eval_fast import evaluate


def main(args):
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    torch.cuda.set_device(device);dist.init_process_group('nccl',device_id=device)
    size=dist.get_world_size();assert size==2
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cpu=load_trace(args.calibration,device='cpu')
    split=check_trace_split(cpu,load_trace_identity(args.evaluation));assert split['prompt_identity_verified']
    records=cpu.shape[1];assert records%size==0
    start=rank*records//size;stop=(rank+1)*records//size
    prep=time.perf_counter()
    trace=replace(cpu,keys=cpu.keys[:,start:stop].to(device),values=cpu.values[:,start:stop].to(device),records=cpu.records[start:stop])
    torch.cuda.synchronize();transfer=time.perf_counter()-prep
    single=read_json(Path(args.single)/'summary.json');results={}
    for bits in (2,4):
        folder=out/f'int{bits}';folder.mkdir(exist_ok=True)
        c=load_config(Path(args.original)/f'int{bits}/params/production/codec.json')
        choice=single['by_bits'][f'int{bits}']['selected']
        prep=time.perf_counter();k,v=unfitted_pair(cpu,c)
        dq=None if choice['mode']=='packed' else make_roundtrip(c,choice['mode']=='compiled')
        model=Fast(trace,c,k,v,device,choice['batch'],30,dq=dq)
        model.pass_data(backward=True)
        for p in model.parameters:p.grad=None
        torch.cuda.synchronize();prepare=time.perf_counter()-prep
        optimizer=torch.optim.Adam(model.parameters,lr=.001)
        history=[];best_loss=float('inf');best=None;best_step=0
        dist.barrier();began=time.perf_counter()
        for step in range(args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            local=model.pass_data(backward=step<args.steps)['residual_quant_mse']
            scalar=torch.tensor(local,dtype=torch.float64,device=device)
            dist.all_reduce(scalar);loss=float(scalar/size)
            if loss<best_loss:
                best_loss=loss;best_step=step;best=[p.detach().clone() for p in model.parameters]
            history.append(dict(step=step,residual_quant_mse=loss))
            if step==args.steps:break
            flat=torch.cat([p.grad.reshape(-1) for p in model.parameters])
            dist.all_reduce(flat);flat.div_(size)
            assert torch.isfinite(flat).all()
            offset=0
            for p in model.parameters:
                p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
            lr=.001*(.1+.9*(1+math.cos(math.pi*step/max(1,args.steps-1)))/2)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();model.project()
            if rank==0 and step%5==0:
                print(json.dumps(dict(event='dp_update',bits=bits,step=step,loss=loss)),flush=True)
                write_json(folder/'progress.json',dict(history=history,elapsed_seconds=time.perf_counter()-began))
        dist.barrier();torch.cuda.synchronize();seconds=time.perf_counter()-began
        times=torch.tensor([seconds,prepare,transfer],dtype=torch.float64,device=device)
        dist.all_reduce(times,op=dist.ReduceOp.MAX);seconds,prepare,transfer=times.tolist()
        with torch.no_grad():
            for p,value in zip(model.parameters,best):p.copy_(value)
        digest=hashlib.sha256()
        for p in model.parameters:digest.update(p.detach().cpu().numpy().tobytes())
        hashes=[None]*size;dist.all_gather_object(hashes,digest.hexdigest());assert len(set(hashes))==1
        if rank==0:
            ko,vo=model.pair();ko,vo=stamp_parameter_identity(ko,vo,signature=c.format_signature(),coordinate_system=k.coordinate_system)
            ko.save(folder/'k_temporal.npz');vo.save(folder/'v_cross_kv.npz');write_json(folder/'codec.json',asdict(c))
            report=dict(status='complete',gpu_count=2,fit_seconds=seconds,prepare_seconds=prepare,transfer_seconds=transfer,
                first_run_from_cpu_data_ready=seconds+prepare+transfer,updates=args.steps,checkpoint_states=args.steps+1,
                full_global_gradient=True,rank_parameter_hashes=hashes,best_step=best_step,history=history,
                initialization='same unfitted previous-frame rule',selected_mode=choice['mode'])
            write_json(folder/'fit_report.json',report)
            test=load_trace(args.evaluation,device='cpu')
            report['heldout']=evaluate(test,c,ko,vo,device,32,8)
            write_json(folder/'result.json',report);results[f'int{bits}']=report
            del test,ko,vo
        dist.barrier()
        del model,optimizer,best;torch.cuda.empty_cache()
    if rank==0:write_json(out/'summary.json',dict(status='complete',by_bits=results))
    dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('calibration','evaluation','original','single','output'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--steps',type=int,default=50)
    main(parser.parse_args())
