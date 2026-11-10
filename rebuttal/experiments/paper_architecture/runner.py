"""Independent paper-aligned reconstruction/storage/decoding experiment.

No model generation. Fit on CPU; wait for an exclusive GPU for measurement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import gc
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from rebuttal.adapters.experimental_codec import ResearchTemporal, ResearchCross, encode_tensor
from rebuttal.common.manifest import file_sha256, write_json, read_json
from rebuttal.common.trace import load_trace, load_trace_identity, check_trace_split
from tempokv.config import load_config
from tempokv.fitting import fit_predictors, fit_cross_kv, _reconstruct_cross_kv_trace
from .storage import storage_bytes
from .temporal import build_temporal, load_temporal, _replay
from .cross import build_cross, load_cross


def specs():
    temporal=[{'name':'paper_channel_fp32','kind':'paper','precision':'fp32'},
              {'name':'copy_previous','kind':'identity','precision':'fp32'},
              {'name':'scalar_fp32','kind':'scalar','precision':'fp32'},
              {'name':'channel_refit_fp32','kind':'channel_refit','precision':'fp32'},
              {'name':'channel_refit_fp16','kind':'channel_refit','precision':'fp16'}]
    temporal += [dict(name=f'dense_{p}',kind='dense',precision=p) for p in ['fp32','fp16','int8','int4']]
    temporal += [dict(name=f'block{b}_{p}',kind='block',block_size=b,precision=p)
                 for b,p in [(2,'fp16'),(4,'fp32'),(16,'fp32'),(32,'fp32'),(64,'fp32')]]
    temporal += [dict(name=f'lowrank_r{r}_{p}',kind='low_rank',rank=r,precision=p)
                 for r,p in [(1,'fp16'),(2,'fp32'),(8,'fp32'),(16,'fp32'),(32,'fp32')]]
    cross=[dict(name='paper_dense_fp32',kind='paper',precision='fp32'),
           dict(name='dense_refit_fp32',kind='dense_refit',precision='fp32')]
    cross += [dict(name=f'dense_{p}',kind='dense_quant',precision=p) for p in ['fp16','int8','int4']]
    cross += [dict(name=f'{kind}_r{r}',kind=kind,rank=r,precision='fp32')
              for kind in ['svd','low_rank_refit'] for r in [8,16,32,48,64]]
    cross += [dict(name='diagonal_refit',kind='diagonal',precision='fp32'),
              dict(name='block32_refit',kind='block',block_size=32,precision='fp32')]
    cross += [dict(name=f'shared_tucker_r{r}',kind='shared_tucker',rank=r,precision='fp32')
              for r in [16,32,64,90,96]]
    history=[dict(name='paper_channel_vhistory',kind='paper',precision='fp32'),
        dict(name='channel_refit_vhistory',kind='block',block_size=1,precision='fp32'),
        dict(name='scalar_vhistory',kind='scalar',precision='fp32'),
        dict(name='dense_vhistory',kind='dense',precision='fp32'),
        dict(name='block2_fp16_vhistory',kind='block',block_size=2,precision='fp16'),
        dict(name='rank1_fp16_vhistory',kind='low_rank',rank=1,precision='fp16')]
    return {'temporal':temporal,'cross':cross,'v_history':history}


def config(path):
    c=read_json(path)
    c['_path']=str(Path(path).resolve())
    return c


def codec(c):
    cfg=load_config(c['codec_config'],c.get('codec_overrides'))
    if (cfg.k_predictor_mode,cfg.cross_kv_weight_mode,cfg.fit_mode)!=('layer_head','layer_head','native'):
        raise ValueError('paper baseline requires independent layer/head native predictors')
    if (cfg.k_history_length,cfg.v_history_length,cfg.k_bits,cfg.v_bits)!=(2,1,4,4):
        raise ValueError('primary paper/old-table profile is KH2/VH1 K4/V4')
    return replace(cfg,use_triton=False,strict_triton=False,cross_matmul_precision='ieee')


def mmap_array(path,name):
    """Read uncompressed pooled NPZ one layer at a time, not34GB of copies."""
    with zipfile.ZipFile(path) as archive:
        item=archive.getinfo(name+'.npy')
        if item.compress_type!=zipfile.ZIP_STORED:
            raise ValueError('formal pooled trace must be uncompressed NPZ for bounded-memory access')
    with Path(path).open('rb') as stream:
        stream.seek(item.header_offset)
        header=stream.read(30);filename_length,extra_length=struct.unpack('<HH',header[26:30])
        stream.seek(filename_length+extra_length,1)
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('unsupported NPY version')
        offset=stream.tell()
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,order='F' if fortran else 'C')


class LayerArray:
    def __init__(self,array):self.array=array
    def __getitem__(self,index):return torch.from_numpy(np.array(self.array[index],dtype=np.float32,copy=True))


class LayerTrace:
    def __init__(self,path):
        k=mmap_array(path,'keys');v=mmap_array(path,'values')
        if k.shape!=v.shape:raise ValueError('traceK/V shape mismatch')
        self.shape=k.shape;self.keys=LayerArray(k);self.values=LayerArray(v)


def prepare(c,out):
    destination=out/'calibration_8p_2x24.npz'
    record=out/'data.json'
    if record.exists():
        d=read_json(record)
        if destination.is_file() and file_sha256(destination)==d['calibration_sha256']:
            return d
        raise ValueError('prepared calibration bytes changed')
    trace=load_trace_identity(c['calibration_trace'])
    evaluation=load_trace_identity(c['evaluation_trace'])
    split=check_trace_split(trace,evaluation)
    if not split['prompt_identity_verified']:
        raise ValueError('paper run requires prompt-disjoint recorded identities')
    sidecar=Path(trace.metadata.get('identity_path',trace.path.with_suffix('.identity.json')))
    records=[{} for _ in range(trace.metadata['original_shape'][1])]
    for row in read_json(sidecar)['records']:
        begin=row.get('record_index',row.get('record_start',0));end=row.get('record_stop',begin+1)
        for i in range(begin,end):records[i]=row
    selected=[]
    for r in records:
        identity=(str(r.get('prompt_id')),r.get('prompt_sha256'))
        if identity not in selected:selected.append(identity)
    selected=selected[:8]
    ids=[i for i,r in enumerate(records) if (str(r.get('prompt_id')),r.get('prompt_sha256')) in selected]
    source_k=mmap_array(trace.path,'keys');source_v=mmap_array(trace.path,'values')
    if len(selected)!=8 or len(ids)!=64 or source_k.shape[2]<48:
        raise ValueError('need8 calibration prompts x8 recorded positions and48 real frames')
    if any(sum((str(records[i].get('prompt_id')),records[i].get('prompt_sha256'))==identity for i in ids)!=8 for identity in selected):
        raise ValueError('each calibration prompt must have exactly8 spatial tracks')
    keys=np.concatenate([source_k[:,ids,:24],source_k[:,ids,24:48]],axis=1)
    values=np.concatenate([source_v[:,ids,:24],source_v[:,ids,24:48]],axis=1)
    np.savez(destination,keys=keys,values=values,layer_ids=np.asarray(trace.metadata['layer_ids']),
             stride=np.asarray(1560),coordinate_system=np.asarray('pre_rope_normalized'))
    d={'status':'complete','calibration_source':str(trace.path),'calibration_source_sha256':file_sha256(trace.path),
       'calibration_sha256':file_sha256(destination),'selected_prompt_ids':[p[0] for p in selected],
       'selected_prompt_hashes':[p[1] for p in selected],'selected_record_ids':ids,'shape':list(keys.shape),
       'disk_dtype':'float32; activations recorded fromBF16 model','tensor_bytes':keys.nbytes+values.nbytes,
       'calibration_protocol':'8prompts,8spatial tracks,2nonoverlapping real24frame clips; clip length follows existing collector, not a manuscript-specified duration',
       'evaluation_source':str(evaluation.path),'evaluation_sha256':file_sha256(evaluation.path),
       'evaluation_prompt_count':len(evaluation.metadata['split_identity']['prompt_hashes']),
       'evaluation_policy':'complete180frame causal feedback; only frame0 anchor; no periodic resets','split':split}
    write_json(record,d)
    return d


def models_for_layer(temporal,cross,layer,k,v,cfg):
    kp,kr,ks=encode_tensor(k,temporal,replace(cfg,stride=1),bits=cfg.k_bits,layer=layer,diagnostics=True)
    vp,vr,vs=encode_tensor(v,cross,replace(cfg,stride=1),bits=cfg.v_bits,layer=layer,source=kr,diagnostics=True)
    return kp,vp,kr,vr,ks,vs


def refit_v(keys,values,cfg,layer_ids):
    cross=fit_cross_kv(keys[:,:,1:],values[:,:,1:],ridge=.1,stride=cfg.stride,layer_ids=layer_ids)
    zero=torch.zeros(keys.shape[0],1,keys.shape[-2],keys.shape[-1],device=keys.device)
    history=_reconstruct_cross_kv_trace(keys,values,replace(cross,history_weights=zero),cfg)
    for iteration in range(2):
        if iteration:history=_reconstruct_cross_kv_trace(keys,values,cross,cfg)
        cross=fit_cross_kv(keys,values,reconstructed_values=history,history_length=1,
            history_mode='diagonal',ridge=.1,stride=cfg.stride,layer_ids=layer_ids)
    return cross


def implementation():
    return {p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}


def fit(c,out):
    data=prepare(c,out)
    cfg=codec(c);models=out/'models';models.mkdir(exist_ok=True)
    baseline_path=out/'paper_baseline.json'
    train=load_trace(out/'calibration_8p_2x24.npz',device='cpu')
    if baseline_path.exists():
        baseline=read_json(baseline_path)
        if baseline['calibration_sha256']!=data['calibration_sha256'] or baseline['codec']!=asdict(cfg):
            raise ValueError('baseline calibration or codec changed')
        for name in ['k_temporal.npz','v_cross_kv.npz']:
            if file_sha256(out/name)!=baseline['parameters'][name]:raise ValueError('paper baseline bytes changed')
    else:
        k,v=fit_predictors(train.keys,train.values,config=cfg,iterations=2,temporal_ridge=.1,
            cross_ridge=.1,v_history_ridge=0.,v_history_mode='diagonal',max_gain=1.,
            fit_history_source='reconstructed',layer_ids=train.layer_ids,coordinate_system='pre_rope_normalized')
        # Store tensors plus primitive metadata; no arbitrary Python class pickle.
        k.save(out/'k_temporal.npz');v.save(out/'v_cross_kv.npz')
        write_json(baseline_path,{'codec':asdict(cfg),'calibration_sha256':data['calibration_sha256'],
            'parameters':{n:file_sha256(out/n) for n in ['k_temporal.npz','v_cross_kv.npz']}})
    from tempokv.parameters import TemporalParameters,CrossKVParameters
    k=TemporalParameters.load(out/'k_temporal.npz');v=CrossKVParameters.load(out/'v_cross_kv.npz')
    kt=ResearchTemporal.from_production(k,len(train.layer_ids),train.keys.shape[-2])
    cv=ResearchCross.from_production(v,'layer_head',len(train.layer_ids),train.keys.shape[-2])
    baseline_k,baseline_v=build_temporal(train.keys,kt,cfg,{'kind':'paper'} )[0],build_cross(train.keys,train.values,cv,cfg,{'kind':'paper'})[0]
    if (baseline_k.layers,baseline_k.heads,baseline_k.dimension)==(30,12,128):
        assert storage_bytes(baseline_k.storage_tensors())==276480*4
        assert storage_bytes(baseline_v.storage_tensors())==5990400*4
    rebuilt_paper_k=_replay(train.keys,baseline_k,cfg)
    rows=[]
    for axis,variants in specs().items():
        for spec in variants:
            path=models/f'{axis}_{spec["name"]}.pt'
            if path.exists():
                row=read_json(path.with_suffix('.json'))
                if file_sha256(path)!=row['sha256']:raise ValueError('model artifact changed')
                rows.append(row);continue
            started=time.perf_counter()
            if axis=='temporal':
                tm,report=build_temporal(train.keys,kt,cfg,spec,rounds=2,ridge=.1,max_gain=1.)
                rebuilt=_replay(train.keys,tm,cfg)
                # Same dense+VH1 family, recalibrated for this reconstructed K.
                if spec['kind']=='paper':vm=baseline_v
                else:
                    vfit=refit_v(rebuilt,train.values,cfg,train.layer_ids)
                    vbase=ResearchCross.from_production(vfit,'layer_head',tm.layers,tm.heads)
                    vm=build_cross(rebuilt,train.values,vbase,cfg,{'kind':'paper'})[0]
            elif axis=='cross':
                tm=baseline_k
                vm,report=build_cross(rebuilt_paper_k,train.values,cv,cfg,spec,rounds=2,ridge=.1)
            else:
                from .history import build_history
                tm=baseline_k
                vm,report=build_history(rebuilt_paper_k,train.values,baseline_v,cfg,spec,rounds=2,ridge=.1)
            state={'temporal':tm.state(),'cross':vm.state(),'spec':spec,'axis':axis,'codec':asdict(cfg)}
            torch.save(state,path)
            kt_bytes=storage_bytes(tm.storage_tensors());vt_bytes=storage_bytes(vm.storage_tensors())
            matrix_bytes=storage_bytes([t for w in vm.weights+vm.factors for t in w.tensors()])
            row={'axis':axis,'name':spec['name'],'spec':spec,'path':str(path),'sha256':file_sha256(path),
                 'serialized_model_bytes':path.stat().st_size,
                 'fit_seconds':time.perf_counter()-started,'fit':report,'k_parameter_bytes':kt_bytes,
                 'v_parameter_bytes':vt_bytes,'cross_matrix_bytes':matrix_bytes,'parameter_bytes':kt_bytes+vt_bytes,
                 'component_parameter_bytes':kt_bytes if axis=='temporal' else (matrix_bytes if axis=='cross' else
                    storage_bytes([t for part in vm.history_parts for t in part.tensors()]) if hasattr(vm,'history_parts') else storage_bytes([vm.history_weights])),
                 'v_family_policy':'dense+VH1 refitted for changed K' if axis=='temporal' else
                    ('paper K and VH1 fixed' if axis=='cross' else 'paper K and dense W fixed; only V-history structure recalibrated'),
                 'initialization_and_constraint_note':'nonpaper K models share persistence bootstrap and incomingL1 cap1; full paper parameters are retained as a separate exact control'}
            write_json(path.with_suffix('.json'),row);rows.append(row)
            print(json.dumps({'fitted':spec['name'],'axis':axis,'parameter_bytes':row['parameter_bytes']}),flush=True)
            del tm,vm;gc.collect()
    write_json(out/'fit.json',{'status':'complete','data':data,'codec':asdict(cfg),'rows':rows,'implementation':implementation(),'experiment_config':c})


def exclusive_gpu(device):
    if not str(device).startswith('cuda'):return
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
    if not physical.isdigit():raise RuntimeError('exclusive timing requires a single explicit physical GPU')
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    uuid=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    own={os.getpid()}
    status=Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('NSpid:'):own.update(int(v) for v in line.split()[1:])
    observed=[int(line.split(',')[1]) for line in result.splitlines() if line.startswith(uuid)]
    if not observed:exclusive_gpu.prechecked_empty=True
    if getattr(exclusive_gpu,'driver_pid',None) is not None:own.add(exclusive_gpu.driver_pid)
    if len(observed)==1 and observed[0] not in own and torch.cuda.is_initialized() and getattr(exclusive_gpu,'prechecked_empty',False):
        exclusive_gpu.driver_pid=observed[0];own.add(observed[0])
    foreign=[pid for pid in observed if pid not in own]
    if foreign:raise RuntimeError('timing GPU has a foreign process; no contaminated timing accepted')


def measure(c,out,device):
    exclusive_gpu(device)
    fit_report=read_json(out/'fit.json');cfg=codec(c)
    if fit_report['codec']!=asdict(cfg):raise ValueError('codec changed between calibration and measurement')
    if fit_report['implementation']!=implementation():raise ValueError('implementation changed after fit')
    if file_sha256(c['evaluation_trace'])!=fit_report['data']['evaluation_sha256']:raise ValueError('evaluation bytes changed')
    trace=LayerTrace(c['evaluation_trace'])
    if trace.shape[2]!=180:raise ValueError('formal evaluation requires the complete180frame trace')
    resultfile=out/'measurements.json';done=read_json(resultfile) if resultfile.exists() else {'rows':[]}
    completed={r['axis']+'/'+r['name'] for r in done['rows']}
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    for record in fit_report['rows']:
        key=record['axis']+'/'+record['name']
        if key in completed:continue
        if file_sha256(record['path'])!=record['sha256']:raise ValueError('predictor changed after fit')
        state=torch.load(record['path'],weights_only=True,map_location='cpu')
        tm=load_temporal(state['temporal'],device);vm=load_cross(state['cross'],device)
        totals={role:{'reconstruction_sse':0.,'target_energy':0.,'prediction_sse':0.,'quantization_sse':0.,'count':0} for role in ['k','v']}
        samples_k=[0.]*int(c.get('decode_repeats',5));samples_v=[0.]*len(samples_k)
        for layer in range(trace.shape[0]):
            exclusive_gpu(device)
            k=trace.keys[layer].permute(0,2,1,3).to(device);v=trace.values[layer].permute(0,2,1,3).to(device)
            kp,vp,kr,vr,ks,vs=models_for_layer(tm,vm,layer,k,v,cfg)
            if not torch.equal(kr,kp.decode()) or not torch.equal(vr,vp.decode(source=kr)):
                raise RuntimeError('encoded reconstruction and actual decoder differ')
            for role,stats in [('k',ks),('v',vs)]:
                for frame in stats:
                    for field in totals[role]:totals[role][field]+=frame[field]
            for iteration in range(int(c.get('decode_warmup',2))+len(samples_k)):
                sync();start=time.perf_counter();dk=kp.decode();sync();middle=time.perf_counter()
                dv=vp.decode(source=dk);sync();end=time.perf_counter()
                if iteration>=int(c.get('decode_warmup',2)):
                    i=iteration-int(c.get('decode_warmup',2));samples_k[i]+=middle-start;samples_v[i]+=end-middle
                del dk,dv
            exclusive_gpu(device)
            del k,v,kp,vp,kr,vr;gc.collect()
        for role in totals:
            totals[role]['reconstruction_nmse']=totals[role]['reconstruction_sse']/totals[role]['target_energy']
        row={**record,'metrics':totals,'decode_k_seconds':samples_k,'decode_v_seconds':samples_v,
             'decode_total_seconds':[a+b for a,b in zip(samples_k,samples_v)],
             'decode_seconds_median':statistics.median(a+b for a,b in zip(samples_k,samples_v)),
             'backend':'common_pytorch_reference','precision_policy':'FP32 IEEE math; stored model precision as declared',
             'timing_scope':'sum over30layer KthenV full180frame decode; weight dequantization included; encode/fit/IO excluded',
             'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_name':torch.cuda.get_device_name() if str(device).startswith('cuda') else 'CPU'}
        done['rows'].append(row);write_json(resultfile,done)
        print(json.dumps({'measured':key,'K_NMSE':totals['k']['reconstruction_nmse'],'V_NMSE':totals['v']['reconstruction_nmse'],'decode_s':row['decode_seconds_median']}),flush=True)
        del tm,vm;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
    summarize(out,done)
    if str(device).startswith('cuda'):
        try:
            measure_production(c,out,device,trace)
        except Exception as error:
            write_json(out/'production_backend.json',{'status':'unavailable','error':repr(error),
                'scope':'additional production Triton control; common-backend architecture measurements are retained'})


def measure_production(c,out,device,trace):
    from tempokv.parameters import TemporalParameters,CrossKVParameters
    from tempokv.fitting import stamp_parameter_identity
    from tempokv.codec import TempoKVCodec,decode_tensor
    cfg=replace(codec(c),stride=1,use_triton=True,strict_triton=True,cross_matmul_precision='tf32')
    k=replace(TemporalParameters.load(out/'k_temporal.npz'),stride=1)
    v=replace(CrossKVParameters.load(out/'v_cross_kv.npz'),stride=1)
    k,v=stamp_parameter_identity(k,v,signature=cfg.format_signature(),coordinate_system='pre_rope_normalized')
    native=TempoKVCodec(k.to(device),v.to(device),cfg)
    durations=[0.]*int(c.get('decode_repeats',5))
    for layer in range(trace.shape[0]):
        exclusive_gpu(device)
        key=trace.keys[layer].permute(0,2,1,3).to(device)
        value=trace.values[layer].permute(0,2,1,3).to(device)
        packed=native.encode(key,value,layer=layer)
        for iteration in range(int(c.get('decode_warmup',2))+len(durations)):
            torch.cuda.synchronize();start=time.perf_counter()
            kr,vr=native.decode(packed)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if iteration>=int(c.get('decode_warmup',2)):durations[iteration-int(c.get('decode_warmup',2))]+=elapsed
            del kr,vr
        exclusive_gpu(device)
        del key,value,packed
    write_json(out/'production_backend.json',{'status':'complete','backend':'production_triton',
        'decode_total_seconds':durations,'median_seconds':statistics.median(durations),
        'math_policy':'production TF32; separate implementation control',
        'comparison_policy':'never attribute PyTorch-versus-Triton speed differences to model architecture',
        'workload':'same recorded180frame token tracks, not end-to-end video generation'})


def summarize(out,measurements):
    rows=measurements['rows'];flat=[];groups=[]
    for r in rows:
        baseline=next(x for x in rows if x['axis']==r['axis'] and x['spec']['kind']=='paper')
        total_baseline=baseline['decode_seconds_median']
        flat.append({'axis':r['axis'],'variant':r['name'],'component_mib':r['component_parameter_bytes']/2**20,
            'total_predictor_mib':r['parameter_bytes']/2**20,'K_NMSE':r['metrics']['k']['reconstruction_nmse'],
            'V_NMSE':r['metrics']['v']['reconstruction_nmse'],'decode_K_ms':statistics.median(r['decode_k_seconds'])*1000,
            'decode_V_ms':statistics.median(r['decode_v_seconds'])*1000,'decode_KV_ms':r['decode_seconds_median']*1000,
            'decode_overhead_vs_paper_pct':100*(r['decode_seconds_median']/total_baseline-1)})
    for axis in ['temporal','cross','v_history']:
        candidates=[r for r in rows if r['axis']==axis]
        for budget in sorted({r['component_parameter_bytes'] for r in candidates if r['component_parameter_bytes']>0}):
            strict=[r['name'] for r in candidates if .95*budget<=r['component_parameter_bytes']<=budget]
            if len(strict)>1:groups.append({'axis':axis,'budget_bytes':budget,'matched_tolerance':.05,'variants':strict,
                'actual_bytes':{r['name']:r['component_parameter_bytes'] for r in candidates if r['name'] in strict},
                'policy':'at most this budget, actual bytes and unused headroom retained; no dummy padding'})
    write_json(out/'budget_groups.json',groups)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write_json(out/'completion.json',{'status':'complete','required_measurements':['closed_loop_reconstruction','actual_parameter_storage','decode_overhead'],
        'variant_count':len(rows),'budget_group_count':len(groups),'one_anchor':True,'layer_head_independent_baseline':True,
        'backend_caveat':'reference implementation architecture comparison; not production Triton speed claims'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['plan','fit','measure'],required=True);parser.add_argument('--device',default='cuda:0')
    a=parser.parse_args();c=config(a.config);out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(int(c.get('threads',4)));torch.backends.cuda.matmul.allow_tf32=False
    if a.stage=='plan':write_json(out/'plan.json',{'config':c,'variants':specs(),'status':'planned','paper_baseline':asdict(codec(c))})
    elif a.stage=='fit':fit(c,out)
    else:measure(c,out,a.device)


if __name__=='__main__':main()
