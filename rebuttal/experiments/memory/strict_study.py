"""Experiment6 under a verified whole-device ceiling, QVG4 and TempoKV K5/V3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256


def environment(out,gpu):
    proof=read_json(out/'hard_memory_verification.json')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid',
        '--format=csv,noheader'],text=True).strip()
    return {**os.environ,'CUDA_VISIBLE_DEVICES':uuid,
        'CUDA_MPS_PIPE_DIRECTORY':proof['pipe_directory'],
        'CUDA_MPS_LOG_DIRECTORY':'/tmp/tempokv_exp6_strict48_mps_20261109/log',
        'TEMPOKV_STRICT_MEMORY_PROOF':str(out/'hard_memory_verification.json'),
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}


def prepare(out,source):
    from rebuttal.experiments.memory.capacity import capacity_report
    from tempokv.config import load_config
    original=read_json(source/'generation_plan.json')
    template=next(j for j in original['jobs'] if j['comparison']=='matched_memory' and j['method']=='tempokv')
    worker=deepcopy(template['worker_config'])
    cfg=load_config(worker['codec_config'],{**worker['codec_overrides'],'k_bits':5,'v_bits':3})
    params=out/'params';params.mkdir(exist_ok=True)
    write_json(params/'codec.json',asdict(cfg))
    # Conservative KV subbudget leaves model, attention and VAE room under the
    # CUDA allocation ceiling. This is not a maximum whole-device capacity claim.
    cap=capacity_report(dict(codec_overrides=asdict(cfg),budgets_bytes=[18*2**30],frames=3837,
        horizon_latent_frames=1488,span_frames=12,bf16_lag_frames=18,
        qvg={**worker['qvg'],'bits':4},methods=['bf16','qvg','tempokv']))
    write_json(out/'capacity_kv_subbudget.json',cap)
    histories={r['method']:r['history_frames'] for r in cap['rows']}
    common=min(96,min(histories.values()))//12*12
    conditions={}
    logical=[]
    for comparison,method,h in [*[( 'matched_memory',m,histories[m]) for m in histories],
            ('tempokv_short','tempokv',histories['bf16']),
            *[('matched_context',m,common) for m in histories]]:
        key=f'{method}_h{h}'
        conditions.setdefault(key,dict(condition=key,method=method,history_frames=h))
        logical.append(dict(comparison=comparison,method=method,history_frames=h,condition=key))
    config=dict(frames=3837,expected_frames=3837,pilot_n=3,seed=0,
        prompts='/root/workspace/Tempokv/results/fixed_memory/prompts_pilot3.txt',
        calibration='/root/workspace/Tempokv/results/architecture/traces/calibration.npz',
        whole_device_limit_bytes=48_000_000_000,cuda_limit_mib=40000,torch_limit_mib=39000,
        kv_subbudget_bytes=18*2**30,shared_history_frames=common,
        checkpoint='/root/workspace/video_models/Self-Forcing/checkpoints/self_forcing_dmd.pt',
        model_config=str(ROOT/'experiments/Self-Forcing/configs/self_forcing_dmd.yaml'),
        conditions=list(conditions.values()),logical_comparisons=logical,
        original_worker=worker,original_command=template['command'],
        source_old_run=str(source),qvg_bits=4,tempokv_k_bits=5,tempokv_v_bits=3,
        capacity_scope='chosen histories under conservative KV subbudget and whole-device hard ceiling; not maximal whole-device capacity')
    write_json(out/'study_config.json',config)
    return config


def fit(out):
    import torch
    from tempokv.integration import load_codec
    import shutil
    config=read_json(out/'study_config.json')
    sources={}
    for role,name in [('k_params','k_temporal.npz'),('v_params','v_cross_kv.npz')]:
        source=Path(config['original_worker'][role])
        destination=out/'params'/name
        shutil.copy2(source,destination)
        assert file_sha256(source)==file_sha256(destination)
        sources[role]=dict(path=str(source),sha256=file_sha256(source))
    actual=load_codec(out/'params/codec.json',out/'params/k_temporal.npz',out/'params/v_cross_kv.npz')
    key=torch.randn(1,12,12*1560,128,device='cuda',dtype=torch.bfloat16)
    value=torch.randn_like(key)
    pair=actual.encode(key,value,layer=0)
    rebuilt=actual.decode(pair)
    assert rebuilt[0].shape==key.shape and rebuilt[1].shape==value.shape
    assert torch.isfinite(rebuilt[0]).all() and torch.isfinite(rebuilt[1]).all()
    write_json(out/'params/completion.json',dict(status='complete',k_bits=5,v_bits=3,
        predictor_policy='unchanged original Eq5 parameters; only quantization bit widths changed',
        source_parameters=sources,real_cuda_codec_roundtrip='passed',
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def sample_job(out,config,condition,pid):
    folder=out/'samples'/condition['condition']/f'prompt_{pid}'
    folder.mkdir(parents=True,exist_ok=True)
    worker=deepcopy(config['original_worker'])
    worker.update(method=condition['method'],history_frames=condition['history_frames'],
        budget_bytes=config['kv_subbudget_bytes'],memory_events=str(folder/'memory_events.jsonl'),
        memory_report=str(folder/'memory_report.json'))
    worker['qvg']={**worker['qvg'],'bits':4}
    if condition['method']=='tempokv':
        worker.update(codec_config=str(out/'params/codec.json'),codec_overrides=read_json(out/'params/codec.json'),
            k_params=str(out/'params/k_temporal.npz'),v_params=str(out/'params/v_cross_kv.npz'))
    write_json(folder/'worker_config.json',worker)
    cmd=[sys.executable,str(ROOT/'rebuttal/experiments/memory/worker_strict.py'),
        '--config',str(folder/'worker_config.json'),'--','--config_path',config['model_config'],
        '--checkpoint_path',config['checkpoint'],'--data_path',config['prompts'],
        '--output_folder',str(folder/'videos'),'--num_output_frames','960',
        '--local_attn_size',str(condition['history_frames']),'--sink_size','0',
        '--seed','0','--num_samples','1','--save_with_index','--use_ema',
        '--prompt_indices',json.dumps([pid]),'--skip_existing']
    return dict(**condition,prompt_id=pid,folder=str(folder),command=cmd)


def validate(job):
    import av
    from rebuttal.experiments.memory.identity import video_identities
    folder=Path(job['folder']);pid=job['prompt_id']
    records,source=video_identities(folder/'videos',prompts_path=Path(read_json(folder.parents[2]/'study_config.json')['prompts']))
    assert len(records)==1 and records[0]['prompt_id']==str(pid)
    assert records[0]['noise_seed']==pid*1_000_003 and records[0]['sample_id']==0
    path=Path(records[0]['path'])
    with av.open(str(path)) as container:
        assert float(container.streams.video[0].average_rate)==16
        frames=sum(1 for _ in container.decode(video=0))
    assert frames==3837
    report=read_json(folder/'memory_report.json')
    assert report['peak_checked_bytes']<=18*2**30
    write_json(folder/'completion.json',dict(status='complete',frames=frames,sha256=file_sha256(path),
        identity=records[0],source_summary=source,kv_report=report))


def aggregate(out,config):
    from rebuttal.experiments.memory.identity import save_job_identities
    jobs=[]
    for condition in config['conditions']:
        folder=out/'generation'/condition['condition'];videos=folder/'videos';videos.mkdir(parents=True,exist_ok=True)
        summaries=[];reports=[]
        for pid in range(3):
            source=out/'samples'/condition['condition']/f'prompt_{pid}'
            assert read_json(source/'completion.json')['status']=='complete'
            destination=videos/f'{pid}_0.mp4'
            if not destination.exists():os.link(source/f'videos/{pid}_0.mp4',destination)
            summaries.extend(read_json(source/'videos/summary.json')['rank_summaries'])
            reports.append(read_json(source/'memory_report.json'))
        write_json(videos/'summary.json',dict(rank_summaries=summaries,
            summary_scope='concatenated actual summaries from three separately generated prompts'))
        cmd=list(config['original_command']);cmd[cmd.index('--data_path')+1]=config['prompts']
        job=dict(**condition,name=condition['condition'],folder=str(folder),command=cmd,
            seed=0,expected_video_count=3,input_digests=dict(prompts=file_sha256(config['prompts'])),status='complete')
        save_job_identities(job)
        write_json(folder/'memory_report.json',dict(measured_peak_kv_bytes=max(r['peak_checked_bytes'] for r in reports),
            reports=reports,whole_device_limit_bytes=48_000_000_000))
        jobs.append(job)
    write_json(out/'generation_plan.json',dict(status='generation_complete',jobs=jobs,
        logical_comparisons=config['logical_comparisons'],unique_video_count=3*len(jobs)))


def quality(out):
    from rebuttal.experiments.memory.identity import video_identities,identity_key
    from rebuttal.experiments.temporal_window import run
    cfg=read_json(out/'study_config.json');h=cfg['shared_history_frames']
    refs={identity_key(r):r for r in video_identities(out/f'generation/bf16_h{h}/videos',prompts_path=Path(cfg['prompts']))[0]}
    pairs=[]
    for method in ('qvg','tempokv'):
        for record in video_identities(out/f'generation/{method}_h{h}/videos',prompts_path=Path(cfg['prompts']))[0]:
            ref=refs[identity_key(record)];assert record['prompt']==ref['prompt']
            pairs.append(dict(reference=ref['path'],candidate=record['path'],prompt_id=record['prompt_id'],
                seed=record['noise_seed'],sample_id=0,model='self_forcing',method=method,
                comparison_id='strict48GB_matched_context'))
    settings=dict(pairs=pairs,expected_frames=3837,metrics=['psnr','ssim','lpips'],
        window_seconds=15,video_decoder='pyav',bootstrap_samples=1000,bootstrap_seed=0,plots=True)
    run(settings,'evaluate',out/'matched_context_quality')


def main(out,source):
    config=prepare(out,source)
    active={};lock=threading.Lock();stop=threading.Event();peaks={};violations=[]
    def audit():
        from rebuttal.experiments.memory.hard_memory_limit import gpu_rows,MIB
        with (out/'device_memory_audit.jsonl').open('a') as handle:
            while not stop.is_set():
                rows=gpu_rows();handle.write(json.dumps(dict(time=time.time(),devices=rows))+'\n');handle.flush()
                for row in rows:
                    used=row['memory_used_mib']*MIB;uuid=row['uuid']
                    peaks[uuid]=max(peaks.get(uuid,0),used)
                    if used>48_000_000_000:
                        violations.append(row)
                        with lock:
                            for process in active.values():
                                try:os.killpg(process.pid,signal.SIGTERM)
                                except ProcessLookupError:pass
                        stop.set()
                stop.wait(.25)
    thread=threading.Thread(target=audit,daemon=True);thread.start()
    def execute(cmd,log,gpu):
        with log.open('a') as handle:
            process=subprocess.Popen(cmd,cwd=ROOT,env=environment(out,gpu),stdout=handle,
                stderr=subprocess.STDOUT,start_new_session=True)
            with lock:active[gpu]=process
            code=process.wait()
            with lock:active.pop(gpu,None)
        if code:raise RuntimeError(f'process failed({code}); see {log}')
    try:
        write_json(out/'status.json',dict(status='running',phase='real_cuda_codec_validation_k5v3',pid=os.getpid()))
        execute([sys.executable,str(ROOT/'rebuttal/experiments/memory/strict_python.py'),__file__,
            '--fit','--output',str(out)],out/'calibration.log',2)
        tasks=queue.Queue()
        for pid in range(3):
            for condition in config['conditions']:
                tasks.put(sample_job(out,config,condition,pid))
        failures=[]
        def lane(gpu):
            while not stop.is_set():
                try:job=tasks.get_nowait()
                except queue.Empty:return
                folder=Path(job['folder']);error=None
                for attempt in range(1,4):
                    try:
                        write_json(folder/'status.json',dict(status='running',gpu=gpu,attempt=attempt,**job))
                        execute(job['command'],folder/f'run_attempt{attempt}.log',gpu)
                        validate(job);error=None;break
                    except Exception:
                        error=traceback.format_exc();time.sleep(5)
                state=dict(status='failed' if error else 'complete',gpu=gpu,**job)
                if error:state['traceback']=error;failures.append(state)
                write_json(folder/'status.json',state)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(lane,gpu) for gpu in range(4)]
            while not all(f.done() for f in futures):
                states=[read_json(p) for p in (out/'samples').glob('*/*/status.json')]
                write_json(out/'status.json',dict(status='running',phase='strict_generation',pid=os.getpid(),
                    expected_videos=3*len(config['conditions']),completed_videos=sum(s['status']=='complete' for s in states),
                    jobs=states,device_peak_bytes=peaks,whole_device_limit_bytes=48_000_000_000))
                time.sleep(10)
            for future in futures:future.result()
        assert not failures and not violations,(failures,violations)
        aggregate(out,config)
        write_json(out/'status.json',dict(status='running',phase='matched_context_quality',pid=os.getpid(),
            generation='complete',device_peak_bytes=peaks))
        execute([sys.executable,str(ROOT/'rebuttal/experiments/memory/strict_python.py'),__file__,
            '--quality','--output',str(out)],out/'quality.log',0)
        write_json(out/'status.json',dict(status='generation_and_quality_complete_vbench_pending',
            device_peak_bytes=peaks,whole_device_limit_bytes=48_000_000_000,
            note='VBench must use strict_python; no unrestricted evaluator permitted'))
    finally:
        stop.set();thread.join(timeout=2)
        write_json(out/'device_memory_summary.json',dict(device_peak_bytes=peaks,
            whole_device_limit_bytes=48_000_000_000,violations=violations,
            enforcement='verified MPS CUDA allocation limit; single experiment client per GPU; NVML audit'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source');parser.add_argument('--output',required=True)
    parser.add_argument('--fit',action='store_true');parser.add_argument('--quality',action='store_true')
    args=parser.parse_args();out=Path(args.output)
    try:
        if args.fit:fit(out)
        elif args.quality:quality(out)
        else:main(out,Path(args.source))
    except Exception:
        write_json(out/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
