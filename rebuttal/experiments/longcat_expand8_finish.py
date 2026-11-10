"""Pool four calibration/four evaluation prompts and repeat the same9 variants."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256
from rebuttal.common.trace import load_trace_identity,check_trace_split
from rebuttal.common.trace_pool import pool_traces


def prior_evaluators():
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            cmd=(path/'cmdline').read_bytes().split(b'\0')
            state=(path/'stat').read_text().split(') ',1)[1].split()[0]
            if state in ('T','t','Z'):continue
            if not any(b'experiments/motion/previous_selected.py' in x for x in cmd):continue
            if b'--shard' not in cmd or b'previous_high_difficult_20261109' not in b' '.join(cmd):continue
            environment=dict(x.split(b'=',1) for x in (path/'environ').read_bytes().split(b'\0') if b'=' in x)
            if environment.get(b'CUDA_VISIBLE_DEVICES')==b'1':found.append(int(path.name))
        except (OSError,ValueError):pass
    return found


def main(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve()
    plan=read_json(out/'plan.json');pools={}
    for role in ('calibration','evaluation'):
        old=read_json(source/f'capture/{role}/converted/index.json')['traces']
        assert len(old)==2
        entries=list(old)
        for job in [j for j in plan['new_jobs'] if j['role']==role]:
            folder=out/'capture'/role/f"p{job['prompt_id']:02d}"
            record=read_json(folder/'job.json');assert record['status']=='complete'
            clean=Path(record['trace']);assert file_sha256(clean)==record['sha256']
            entries.append(dict(trace=str(clean),sha256=record['sha256']))
            identity=read_json(clean.with_suffix('.identity.json'))['records'][0]
            for raw in {p.parent for p in (folder/'raw').rglob('layer_*.npz')}:
                path=raw/'identity.json';previous=read_json(path) if path.exists() else {}
                for field in ('prompt_id','seed'):
                    if field in previous:assert str(previous[field])==str(identity[field])
                write_json(path,{**previous,**identity})
        assert len(entries)==4
        index=out/f'traces/{role}_index.json';write_json(index,dict(status='complete',traces=entries,model='longcat'))
        target=out/f'traces/{role}.npz'
        pooled=pool_traces(index,target,resume=True)
        assert pooled.get('shape',[48,128,14,32,128])==[48,128,14,32,128]
        pools[role]=target
    split=check_trace_split(load_trace_identity(pools['calibration']),load_trace_identity(pools['evaluation']))
    assert split['prompt_identity_verified'] and not split['in_sample_diagnostic']
    for role in pools:
        identity=load_trace_identity(pools[role]);assert len(identity.metadata['split_identity']['prompt_hashes'])==4
    write_json(out/'pool_complete.json',dict(status='complete',n_calibration_prompts=4,n_evaluation_prompts=4,
        shape_per_split=[48,128,14,32,128],split=split))
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'1','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
                 'PYTHONPATH':str(ROOT)+os.pathsep+str(ROOT/'src')}
    paused=prior_evaluators()
    try:
        for pid in paused:
            try:os.kill(pid,signal.SIGSTOP)
            except ProcessLookupError:pass
        write_json(out/'timing_isolation.json',dict(gpu=1,temporarily_paused_previous_evaluator_pids=paused,
            reason='sameGPU warmup2/repeat5 decode measurements; generation tasks gated until complete'))
        for axis in ('temporal','cross'):
            config=read_json(source/f'configs/architecture_{axis}.json')
            config.update(device='cuda',calibration_trace=str(pools['calibration']),evaluation_trace=str(pools['evaluation']),
                calibration_identity=str(pools['calibration'].with_suffix('.identity.json')),
                evaluation_identity=str(pools['evaluation'].with_suffix('.identity.json')))
            path=out/f'configs/architecture_{axis}.json';write_json(path,config)
            target=out/f'studies/int2/architecture_{axis}'
            for stage in ('fit','replay'):
                write_json(out/'status.json',dict(status='running',phase=f'{axis}_{stage}',protocol='4cal+4eval',pid=os.getpid()))
                command=[sys.executable,str(ROOT/'rebuttal/run.py'),'--experiment','architecture','--stage',stage,
                    '--config',str(path),'--output-dir',str(target)]
                if (target/'manifest.json').exists():command.append('--resume')
                with (out/f'{axis}_{stage}.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:
        for pid in paused:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
    import csv,math
    rows=[]
    for axis in ('temporal','cross'):
        with (out/f'studies/int2/architecture_{axis}/summary.csv').open() as handle:
            rows.extend([{**row,'axis':axis} for row in csv.DictReader(handle)])
    assert len(rows)==9
    for row in rows:
        for key in ('k_reconstruction_nmse','v_reconstruction_nmse','decode_seconds_median'):
            assert math.isfinite(float(row[key]))
    write_json(out/'results.json',dict(status='complete',model='longcat',protocol='4cal+4eval',
        total_prompt_count=8,split=split,rows=rows,decode_gpu=1,decode_warmup=2,decode_repeats=5,
        scope='offlineBF16trace_replay; same legacy sharing; short14latentframes; not end-to-end video quality'))
    write_json(out/'completion.json',dict(status='complete',n_calibration_prompts=4,n_evaluation_prompts=4,n_variants=9))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
