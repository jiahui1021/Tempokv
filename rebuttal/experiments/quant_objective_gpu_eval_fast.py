"""Residual-MSE-only GPU evaluation, batching independent spans.

Uses the codec prediction/quantization arithmetic and BF16 reconstructed history.
Only final aggregate scalars cross from GPU to CPU, with no event generation.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from tempokv.codec import anchor_roundtrip,predict_temporal,predict_cross_kv,_quantize
from tempokv.config import load_config
from rebuttal.common.trace import load_trace,load_trace_identity,check_trace_split
from rebuttal.common.cache_metrics import load_pair
from rebuttal.common.manifest import write_json,file_sha256


@torch.no_grad()
def span_sums(keys,values,config,k,v,layer_start):
    """[L,N,F,H,D] spans; all N histories are independent."""
    layers,records,frames,heads,width=keys.shape
    assert not k.layered and config.cross_kv_weight_mode=='layer_shared'
    projection=v.for_layer(v.layer_ids[layer_start])
    assert torch.equal(v.weight,v.weight[:1].expand_as(v.weight))
    assert torch.equal(v.bias,v.bias[:1].expand_as(v.bias))
    alpha=v.history_weights[layer_start:layer_start+layers]
    kh,vh=[],[]
    totals=torch.zeros(4,dtype=torch.float64,device=keys.device)
    for frame in range(frames):
        key=keys[:,:,frame].permute(0,2,1,3)
        value=values[:,:,frame].permute(0,2,1,3)
        if frame==0:
            kr=anchor_roundtrip(key,config).to(torch.bfloat16).float()
            vr=anchor_roundtrip(value,config).to(torch.bfloat16).float()
        else:
            kp=predict_temporal(kh,k,legacy=True)
            rk=key-kp
            qk=_quantize(rk,bits=config.k_bits,group_size=config.group_size,
                         mode=config.residual_mode,scale_dtype=getattr(torch,config.scale_dtype))[3]
            kr=(kp+qk).to(torch.bfloat16).float()
            vp=predict_cross_kv(kr,projection,(),legacy=True,
                                matmul_precision=config.cross_matmul_precision)
            for lag,previous in enumerate(vh[:config.v_history_length]):
                vp=vp+alpha[:,lag,:,None,:]*previous
            rv=value-vp
            qv=_quantize(rv,bits=config.v_bits,group_size=config.group_size,
                         mode=config.residual_mode,scale_dtype=getattr(torch,config.scale_dtype))[3]
            vr=(vp+qv).to(torch.bfloat16).float()
            totals+=torch.stack(((rk-qk).double().square().sum(),
                                  (rv-qv).double().square().sum(),
                                  key.double().square().sum(),value.double().square().sum()))
        kh=[kr,*kh[:max(k.history_length-1,0)]]
        vh=[vr,*vh[:max(config.v_history_length-1,0)]]
    count=layers*records*(frames-1)*heads*width
    return totals,count


def evaluate(trace,config,k,v,device,batch_records=32,layer_batch=8):
    assert config.fit_mode=='legacy' and config.k_predictor_mode=='shared'
    assert config.v_predictor_mode=='cross_kv'
    assert config.reconstruction_dtype=='bfloat16'
    total=torch.zeros(4,dtype=torch.float64,device=device);count=0
    layers,records,frames,heads,width=trace.shape
    for layer in range(0,layers,layer_batch):
        stop=min(layer+layer_batch,layers)
        for start in range(0,records,batch_records):
            end=min(start+batch_records,records)
            keys=trace.keys[layer:stop,start:end].to(device)
            values=trace.values[layer:stop,start:end].to(device)
            full=frames//24
            if full:
                shape=(stop-layer,end-start,full,24,heads,width)
                kk=keys[:,:,:full*24].reshape(shape).flatten(1,2)
                vv=values[:,:,:full*24].reshape(shape).flatten(1,2)
                stats,n=span_sums(kk,vv,config,k,v,layer)
                total+=stats;count+=n
                del kk,vv
            if frames%24:
                stats,n=span_sums(keys[:,:,full*24:],values[:,:,full*24:],config,k,v,layer)
                total+=stats;count+=n
            del keys,values
        print(f'layers {layer}:{stop} complete',flush=True)
    sums=total.cpu().tolist()
    return {role:{'non_anchor':dict(numel=count,quant_error_sse=sums[index],
                signal_sse=sums[index+2],residual_quant_mse=sums[index]/count,
                residual_quant_nmse_x=sums[index]/sums[index+2])}
            for index,role in enumerate(('k','v'))}


def run(args):
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.06,device)
    trace=load_trace(args.evaluation,device='cpu')
    split=check_trace_split(load_trace_identity(args.calibration),trace)
    assert split['prompt_identity_verified']
    test_hash=file_sha256(trace.path)
    root=Path(args.fit_root);assert (root/'fit_complete.json').is_file()
    for arm,folder in dict(original=Path(args.original),prediction=root/'prediction',quantization=root/'quantization').items():
        config=load_config(folder/'codec.json')
        k,v=load_pair(folder/'k_temporal.npz',folder/'v_cross_kv.npz',config,device)
        began=time.perf_counter()
        metrics=evaluate(trace,config,k,v,device)
        seconds=time.perf_counter()-began
        result=dict(status='complete',arm=arm,metric_scope='bf16_trace_replay',
            trace=str(trace.path),trace_sha256=test_hash,shape=list(trace.shape),span_frames=24,
            config=asdict(config),trace_split=split,device='GPU',batch_records=32,layer_batch=8,
            independent_spans_batched=True,evaluation_seconds=seconds,**metrics)
        write_json(root/'evaluation_fast'/arm/'summary.json',result)
        print(f'{arm} complete in {seconds:.3f}s',flush=True)
    write_json(root/'evaluation_fast/complete.json',dict(status='complete',test_split=split))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('fit-root','original','calibration','evaluation'):
        parser.add_argument('--'+field,required=True)
    run(parser.parse_args())
