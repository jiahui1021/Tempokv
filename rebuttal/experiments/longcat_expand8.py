"""Append2cal+2eval LongCat prompts using four independent GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256


def sha(text):return hashlib.sha256(text.strip().encode()).hexdigest()


def collect(job,out,args):
    folder=out/'capture'/job['role']/f"p{job['prompt_id']:02d}"
    clean=folder/'clean_trace.npz'
    if clean.exists() and clean.with_suffix('.identity.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTHON_BIN':'/root/workspace/.venvs/tempokv-longcat/bin/python',
        'TORCHRUN_BIN':'/root/workspace/.venvs/tempokv-longcat/bin/torchrun',
        'CHECKPOINT_DIR':'/root/workspace/video_models/LongCat-Video','TRACE_DIR':str(folder),
        'TRACE_FILE':str(clean),'OUTPUT_DIR':str(folder/'output'),'PROMPT':job['prompt'],
        'PROMPT_IDX':str(job['prompt_id']),'SEED':str(job['seed']),'NUM_SEGMENTS':'1','NUM_FRAMES':'93',
        'NUM_COND_FRAMES':'53','NUM_INFERENCE_STEPS':'50','GUIDANCE_SCALE':'4.0',
        'CONTEXT_PARALLEL_SIZE':'1','NUM_GPUS':'1','MAX_TRACE_SPATIAL_POSITIONS':'32',
        'CUDA_VISIBLE_DEVICES':str(job['gpu']),'OVERWRITE':'0','PREFLIGHT':'1',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    write_json(folder/'job.json',dict(status='running',**job))
    with (folder/'collect.log').open('a') as log:
        subprocess.run(['bash','scripts/LongCat/collect_trace.sh'],cwd=args.ready,env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
    import numpy as np
    with np.load(clean,allow_pickle=False) as data:
        shape=list(data['keys'].shape)
        assert shape==[48,32,14,32,128] and data['values'].shape==data['keys'].shape
        assert str(data['coordinate_system'])=='pre_rope_normalized'
        assert int(data['stride'])==1560
    identity=dict(prompt_id=job['prompt_id'],prompt_sha256=sha(job['prompt']),seed=job['seed'],sample_id=0,
        fps=15,pixel_frames=133,model='longcat',num_frames=93,num_cond_frames=53,num_segments=1,
        source_scope='longcat_bf16_clean_trace',prompt_text=job['prompt'],record_start=0,record_stop=32,
        absolute_start_frame=0,token_indices=list(range(32)))
    write_json(clean.with_suffix('.identity.json'),dict(records=[identity],source=str(folder/'raw')))
    for path in (folder/'raw').rglob('layer_*.npz'):
        side=path.parent/'identity.json'
        if not side.exists():write_json(side,identity)
    write_json(folder/'job.json',dict(status='complete',trace=str(clean),shape=shape,sha256=file_sha256(clean),**job))
    print(json.dumps(dict(event='LongCat_trace_complete',**job,shape=shape)),flush=True)


def main(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    prompts={}
    for role,prefix in [('calibration','cal'),('evaluation','eval')]:
        old=(source/f'prompts/{prefix}_2.txt').read_text().splitlines()
        extended=(source/f'prompts/{prefix}_8.txt').read_text().splitlines()
        old=[x.strip() for x in old if x.strip()];extended=[x.strip() for x in extended if x.strip()]
        assert len(old)==2 and len(extended)>=4 and extended[:2]==old
        prompts[role]=extended[:4]
    cal={sha(x) for x in prompts['calibration']};evaluation={sha(x) for x in prompts['evaluation']}
    assert len(cal)==len(evaluation)==4 and not cal&evaluation
    jobs=[]
    for role in ('calibration','evaluation'):
        for index in (2,3):
            jobs.append(dict(role=role,prompt_id=index,prompt=prompts[role][index],
                seed=index+(10000 if role=='evaluation' else 0),gpu=len(jobs)))
    write_json(out/'plan.json',dict(protocol='4cal+4eval',new_jobs=jobs,all_prompts=prompts,
        source=str(source),ready_checkout=str(Path(args.ready).resolve()),collect_script_sha256=file_sha256(Path(args.ready)/'scripts/LongCat/collect_trace.sh'),
        settings=dict(num_frames=93,num_cond_frames=53,num_segments=1,inference_steps=50,guidance_scale=4.,max_spatial_positions=32,context_parallel_size=1),
        original_traces_preserved=True,split_prompt_hashes_disjoint=True))
    write_json(out/'status.json',dict(status='waiting',phase='previous_generation_only',pid=os.getpid(),protocol='4cal+4eval'))
    previous=Path(args.previous)
    while not (previous/'generation_complete.json').exists():
        if read_json(previous/'status.json').get('status')=='failed':raise RuntimeError('previous generation failed')
        time.sleep(15)
    write_json(out/'status.json',dict(status='running',phase='four_gpu_capture',pid=os.getpid(),new_traces=4))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(collect,j,out,args) for j in jobs]:future.result()
    write_json(out/'collection_complete.json',dict(status='complete',new_trace_count=4,total_prompt_count=8))
    subprocess.run([sys.executable,str(ROOT/'rebuttal/experiments/longcat_expand8_finish.py'),
        '--source',str(source),'--output',str(out)],check=True)
    write_json(out/'status.json',dict(status='complete',protocol='4cal+4eval',total_prompt_count=8))
    write_json(out/'gpu_release.json',dict(status='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--previous',required=True);parser.add_argument('--ready',default='/root/workspace/Tempokv-architecture4-ready')
    args=parser.parse_args()
    try:main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
