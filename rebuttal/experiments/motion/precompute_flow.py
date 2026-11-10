"""Use idle GPUs for BF16 flow before paired video generation finishes."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json
from rebuttal.experiments.motion import report
from rebuttal.experiments.motion.expand90_finish import helper,cache_name


def ensure_cache(pair,config,path,backend,original):
    """Publish a validated new cache atomically; never overwrite an existing one."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.prefetch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        recipe=original.recipe(pair,config)
        if path.exists():
            original.validate_cache_recipe(read_json(path),recipe,path);return True
        rows,summary=report.analyze_reference(pair,config,backend)
        temporary=path.with_name(f'.{path.name}.prefetch.{os.getpid()}')
        write_json(temporary,dict(recipe=recipe,flow_rows=rows,motion_summary=summary))
        try:
            try:os.link(temporary,path)
            except FileExistsError:original.validate_cache_recipe(read_json(path),recipe,path)
        finally:temporary.unlink(missing_ok=True)
    return True


def main(args):
    out=Path(args.output).resolve();original=helper(out)
    config=read_json(out.parent/'manifest.json')['config']
    summary=read_json(out/'runs/bf16/videos/summary.json')
    rows=[r for shard in summary['rank_summaries'] for r in shard['videos']]
    assert {r['prompt_index'] for r in rows}==set(range(50,90))
    backend=report.make_flow_backend(config['flow'])
    completed=[];busy=[]
    ordered=sorted(rows,key=lambda r:r['prompt_index'],reverse=True)
    # Start on different videos, then opportunistically handle unlocked work.
    ordered=ordered[args.worker::2]+ordered[1-args.worker::2]
    for row in ordered:
        pid=row['prompt_index']
        pair=dict(model='self_forcing',prompt_id=str(pid),seed=row['noise_seed'],sample_id=row['sample_index'],
            reference=str(out/f'runs/bf16/videos/{pid}_0.mp4'),
            candidate=str(out/f'runs/tempokv/videos/{pid}_0.mp4'),method='tempokv',comparison_id='default')
        path=out/'flow_cache'/cache_name(pair,original)
        write_json(out/f'preflow_{args.worker}.json',dict(status='running',pid=os.getpid(),prompt_id=pid,
            completed_prompt_ids=completed,busy_prompt_ids=busy,updated_at=time.time()))
        if ensure_cache(pair,config,path,backend,original):completed.append(pid)
        else:busy.append(pid)
        print(f'preflow worker{args.worker} completed/cache-validated {pid}={pid in completed}',flush=True)
    write_json(out/f'preflow_{args.worker}.json',dict(status='complete',pid=os.getpid(),completed_prompt_ids=completed,
        busy_prompt_ids=busy,updated_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--worker',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        write_json(Path(args.output)/f'preflow_{args.worker}.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
