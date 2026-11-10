"""Extra independent flow workers to overlap CPU stages with small GPU jobs."""
import argparse
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
from rebuttal.experiments.motion.precompute_flow import ensure_cache


def main(args):
    out=Path(args.output).resolve();original=helper(out)
    config=read_json(out.parent/'manifest.json')['config']
    rows=[r for shard in read_json(out/'runs/bf16/videos/summary.json')['rank_summaries'] for r in shard['videos']]
    assert {r['prompt_index'] for r in rows}==set(range(50,90))
    backend=report.make_flow_backend(config['flow'])
    rows=sorted(rows,key=lambda r:r['prompt_index'],reverse=True)
    offset=(args.worker+2)%len(rows);rows=rows[offset:]+rows[:offset]
    completed=[];busy=[];state=out/f'flow_pool_{args.worker}.json'
    for row in rows:
        pid=row['prompt_index']
        pair=dict(model='self_forcing',prompt_id=str(pid),seed=row['noise_seed'],sample_id=row['sample_index'],
            reference=str(out/f'runs/bf16/videos/{pid}_0.mp4'),candidate=str(out/f'runs/tempokv/videos/{pid}_0.mp4'),
            method='tempokv',comparison_id='default')
        path=out/'flow_cache'/cache_name(pair,original)
        write_json(state,dict(status='running',pid=os.getpid(),worker=args.worker,prompt_id=pid,
            completed_prompt_ids=completed,busy_prompt_ids=busy,updated_at=time.time()))
        if ensure_cache(pair,config,path,backend,original):completed.append(pid)
        else:busy.append(pid)
        print(f'pool{args.worker} prompt{pid}: cache_ready={pid in completed}',flush=True)
    write_json(state,dict(status='complete',pid=os.getpid(),worker=args.worker,completed_prompt_ids=completed,
        busy_prompt_ids=busy,updated_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--worker',type=int,required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        write_json(Path(args.output)/f'flow_pool_{args.worker}.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
