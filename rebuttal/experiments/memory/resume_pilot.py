"""Resume the existing three-prompt memory study without overwriting old runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256


def validate_videos(job):
    import av
    videos=Path(job['folder'])/'videos'
    paths=sorted(videos.glob('*.mp4'))
    assert {p.name for p in paths}=={'0_0.mp4','1_0.mp4','2_0.mp4'}
    evidence=[]
    for path in paths:
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            assert float(stream.average_rate)==16
            frames=sum(1 for _ in container.decode(video=0))
        assert frames==3837,(path,frames)
        evidence.append(dict(path=str(path),frames=frames,sha256=file_sha256(path)))
    return evidence


def prepare(source,out):
    assert 'install_cpu_video_output(self.vae.model)' in (ROOT/'rebuttal/experiments/memory/worker.py').read_text()
    original=read_json(source/'generation_plan.json')
    assert original['frames']==3837 and original.get('pilot_n')==3
    assert all(j['expected_video_count']==3 and j['budget_bytes']==48*2**30 for j in original['jobs'])
    prompts=source/'prompts_pilot3.txt'
    assert len(prompts.read_text().splitlines())==3
    first50=Path('/root/workspace/Tempokv/results/_runner/prompts_first50.txt')
    assert first50.read_text().splitlines()[:3]==prompts.read_text().splitlines()
    plan=deepcopy(original)
    for job in plan['jobs']:
        previous=deepcopy(job)
        folder=out/'generation'/job['name']
        folder.mkdir(parents=True)
        worker=job['worker_config']
        for key in ('codec_config','k_params','v_params','qvg_root','model_root'):
            if key in worker:
                worker[key]=str(Path(worker[key]).resolve(strict=True))
        for key,name in [('memory_events','memory_events.jsonl'),('memory_report','memory_report.json')]:
            worker[key]=str(folder/name)
        command=job['command']
        updates={'--config':str(folder/'worker_config.json'),'--data_path':str(prompts),
                 '--output_folder':str(folder/'videos')}
        for flag,value in updates.items():
            command[command.index(flag)+1]=value
        command[0]=sys.executable
        command[1]=str(ROOT/'rebuttal/experiments/memory/worker.py')
        for flag in ('--config_path','--checkpoint_path'):
            idx=command.index(flag)+1
            command[idx]=str(Path(command[idx]).resolve(strict=True))
        job.update(folder=str(folder),config_path=str(folder/'worker_config.json'),status='planned')
        assert file_sha256(prompts)==job['input_digests']['prompts']
        job['identity']=hashlib.sha256(json.dumps({'worker':worker,'command':command,
            'input_digests':job['input_digests']},sort_keys=True).encode()).hexdigest()
        old_videos=Path(previous['folder'])/'videos'
        existing=sorted(old_videos.glob('*.mp4'))
        if existing:
            assert {p.name for p in existing}=={'0_0.mp4','1_0.mp4','2_0.mp4'}
            (folder/'videos').mkdir()
            for video in existing:
                os.link(video,folder/'videos'/video.name)
            shutil.copy2(Path(previous['folder'])/'memory_report.json',folder/'memory_report.json')
            job['reuse_source']=previous['folder']
            write_json(folder/'reuse_provenance.json',dict(source_job=previous,
                scope='reuse original video bytes and KV report; native skip-existing records regenerated',
                whole_device_peak='not measured by resume metadata-only pass'))
        write_json(folder/'worker_config.json',worker)
        write_json(folder/'job.json',job)
        write_json(out/'job_specs'/f"{plan['jobs'].index(job):02d}.json",job)
    plan.update(status='running',source_plan=str(source/'generation_plan.json'),
                source_plan_sha256=file_sha256(source/'generation_plan.json'))
    write_json(out/'generation_plan.json',plan)
    write_json(out/'config.json',dict(prompts=str(prompts),frames=3837,expected_frames=3837,
        metrics=['psnr','ssim','lpips'],window_seconds=15,video_decoder='pyav',
        bootstrap_samples=1000,bootstrap_seed=0,plots=True,
        budget_bytes=48*2**30,pilot_n=3,scope='existing experiment6 pilot3, unchanged controls'))
    return plan


def run_one(args):
    from rebuttal.experiments.memory.generation import execute_jobs
    job=read_json(Path(args.job))
    if job.get('reuse_source'):
        evidence=validate_videos(job)
        write_json(Path(job['folder'])/'reuse_validation.json',dict(videos=evidence,status='complete'))
    execute_jobs(dict(missing_configuration=[],jobs=[job]),resume=True)
    evidence=validate_videos(job)
    report=read_json(Path(job['folder'])/'memory_report.json')
    assert report['peak_checked_bytes']<=job['budget_bytes']
    write_json(Path(job['folder'])/'completion.json',dict(status='complete',videos=evidence,
        budget_bytes=job['budget_bytes'],measured_peak_kv_bytes=report['peak_checked_bytes'],
        history_frames=job['history_frames'],reuse_source=job.get('reuse_source')))


def main(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    plan=prepare(source,out)
    write_json(out/'status.json',dict(status='running',phase='generation',pid=os.getpid(),pilot_n=3))
    ready=queue.Queue()
    def generate(job,gpu):
        index=plan['jobs'].index(job)
        spec=out/'job_specs'/f'{index:02d}.json'
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
             'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
        result=None
        for attempt in range(1,4):
            with (Path(job['folder'])/f'launcher_attempt{attempt}.log').open('a') as log:
                process=subprocess.Popen([sys.executable,__file__,'--job',str(spec)],env=env,
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(Path(job['folder'])/'live_status.json',dict(status='running',pid=process.pid,
                    gpu=gpu,attempt=attempt,job=job['name']))
                code=process.wait()
            if code==0:
                result=dict(status='complete',job=job['name'],gpu=gpu,attempt=attempt)
                break
            time.sleep(5)
        if result is None:
            result=dict(status='failed',job=job['name'],gpu=gpu,attempt=3)
        write_json(Path(job['folder'])/'live_status.json',result)
        if job['comparison']=='matched_memory':
            ready.put((job,result))
        return result
    def quality():
        from rebuttal.experiments.memory.identity import job_video_identities,identity_key
        from rebuttal.experiments.temporal_window import run
        jobs=[j for j in plan['jobs'] if j['comparison']=='matched_context']
        reference=next(j for j in jobs if j['method']=='bf16')
        refs={identity_key(r):r for r in job_video_identities(reference)[0]}
        pairs=[]
        for job in jobs:
            if job['method']=='bf16':
                continue
            for row in job_video_identities(job)[0]:
                ref=refs[identity_key(row)]
                assert ref['prompt']==row['prompt']
                pairs.append(dict(reference=ref['path'],candidate=row['path'],prompt_id=row['prompt_id'],
                    seed=row['noise_seed'],sample_id=row['sample_id'],model='self_forcing',method=job['method'],
                    comparison_id='matched_context_48gib_pilot3'))
        cfg=read_json(out/'config.json');cfg['pairs']=pairs
        return run(cfg,'evaluate',out/'matched_context_quality')
    def vbench(job):
        from rebuttal.experiments.memory.evaluation import evaluate_vbench
        folder=out/'vbench_jobs'/job['name'];folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'generation_plan.json',dict(jobs=[job]))
        deps=source/'dependencies'
        import imageio_ffmpeg
        binary=out/'bin/ffmpeg'
        binary.parent.mkdir(exist_ok=True)
        if not binary.exists():
            binary.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ['PATH']=str(binary.parent)+os.pathsep+os.environ.get('PATH','')
        wrapper=out/'vbench_python.sh'
        if not wrapper.exists():
            wrapper.write_text('#!/bin/bash\nset -euo pipefail\n'
                f'export PYTHONPATH="{deps}/vbench:{deps}/vbench_python:{ROOT}/src${{PYTHONPATH:+:$PYTHONPATH}}"\n'
                f'export VBENCH_CACHE_DIR="{deps}/vbench_cache"\n'
                'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1\n'
                f'exec "{sys.executable}" "$@"\n')
            wrapper.chmod(0o755)
        cfg=read_json(out/'config.json')
        cfg.update(_resume=True,vbench=dict(python=str(wrapper),device='cuda',local_checkpoints=True))
        result=evaluate_vbench(cfg,folder)
        write_json(folder/'completion.json',result)
    def controls_and_metrics():
        os.environ['CUDA_VISIBLE_DEVICES']='3'
        results=[];errors=[]
        controls=[j for j in plan['jobs'] if j['comparison']!='matched_memory']
        for job in controls:
            results.append(generate(job,3))
        if any(r['status']!='complete' for r in results):
            errors.append(dict(stage='control_generation',results=results))
        if all(r['status']=='complete' for r in results):
            try:
                write_json(out/'quality_live_status.json',dict(status='running',gpu=3))
                quality()
                write_json(out/'quality_live_status.json',dict(status='complete',gpu=3))
            except Exception:
                errors.append(dict(stage='matched_context_quality',traceback=traceback.format_exc()))
                write_json(out/'quality_live_status.json',dict(status='failed',traceback=traceback.format_exc()))
        for job in controls:
            if (Path(job['folder'])/'completion.json').exists():
                try:
                    vbench(job)
                except Exception:
                    errors.append(dict(stage='vbench',job=job['name'],traceback=traceback.format_exc()))
                    write_json(out/'metric_errors.json',dict(errors=errors))
        for _ in range(3):
            job,result=ready.get()
            if result['status']=='complete':
                try:
                    vbench(job)
                except Exception:
                    errors.append(dict(stage='vbench',job=job['name'],traceback=traceback.format_exc()))
                    write_json(out/'metric_errors.json',dict(errors=errors))
        write_json(out/'metric_errors.json',dict(errors=errors))
        return dict(status='complete' if not errors else 'failed',errors=errors)
    saturated=[j for j in plan['jobs'] if j['comparison']=='matched_memory']
    assert {j['method'] for j in saturated}=={'bf16','qvg','tempokv'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(generate,job,gpu) for gpu,job in enumerate(saturated)]
        futures.append(pool.submit(controls_and_metrics))
        while not all(f.done() for f in futures):
            states=[]
            for job in plan['jobs']:
                path=Path(job['folder'])/'live_status.json'
                state=read_json(path) if path.exists() else dict(status='pending')
                state.update(job=job['name'],completed_videos=len(list((Path(job['folder'])/'videos').glob('*.mp4'))),
                             expected_videos=3)
                states.append(state)
            write_json(out/'status.json',dict(status='running',phase='generation_and_metrics',pid=os.getpid(),
                updated_unix=time.time(),pilot_n=3,budget_bytes=48*2**30,jobs=states))
            time.sleep(15)
        results=[f.result() for f in futures]
    status='complete' if all(r['status']=='complete' for r in results) else 'incomplete'
    write_json(out/'status.json',dict(status=status,results=results,pid=os.getpid(),pilot_n=3))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source');parser.add_argument('--output');parser.add_argument('--job')
    args=parser.parse_args()
    try:
        (run_one if args.job else main)(args)
    except Exception:
        if args.output:
            write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
