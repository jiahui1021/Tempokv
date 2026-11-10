"""Previous-frame evaluation restricted to high-motion videos and hard windows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json
from rebuttal.common.video_metrics import read_csv,write_csv,iter_pair_metrics,aggregate_frames,make_quality_backend
from rebuttal.adapters.generation import build_job,execute_jobs
from rebuttal.experiments.motion import report

METRICS=('psnr_db','ssim','lpips','k_reconstruction_nmse','v_reconstruction_nmse')


def partition_events(folder,selected,destination):
    by_prompt={str(p):{} for p in selected}
    for path in sorted(folder.glob('cache_events.rank-*.jsonl')):
        with path.open() as handle:
            for line in handle:
                event=json.loads(line);pid=str(event['prompt_id'])
                if pid in by_prompt:
                    key=tuple(str(event[k]) for k in ('seed','sample_id','role','layer','absolute_latent_frame'))
                    by_prompt[pid][key]=line
    destination.mkdir(parents=True,exist_ok=True)
    for pid,events in by_prompt.items():
        assert events,f'missing cache events for {pid}'
        with (destination/f'{pid}.jsonl').open('w') as handle:handle.writelines(events.values())


def evaluate_shard(args):
    base=Path(args.base);out=Path(args.output)
    selection=read_json(out/'selection.json');config=read_json(out/'config.json')
    pairs={str(p['prompt_id']):p for p in read_json(base/args.pairs_file)['pairs']}
    analysis=base/args.analysis_subdir
    cases=[c for c in read_json(analysis/'difficult_cases.json')['cases'] if c['case_id'] in selection['case_ids']]
    tk_video={str(r['prompt_id']):r for r in read_csv(analysis/'motion_videos.csv')}
    assigned=selection['selected_prompt_ids'][args.shard::4]
    if not assigned:
        return
    backend=make_quality_backend(config)
    for pid in assigned:
        key=str(pid);p=pairs[key];high=pid in selection['high_motion_prompt_ids']
        windows=[c for c in cases if str(c['prompt_id'])==key]
        pair={**p,'method':'previous_frame','candidate':str(out/f'runs/previous_frame/videos/{pid}_0.mp4'),
              'cache_events':str(out/f'cache_by_prompt/{pid}.jsonl')}
        if not high:
            pair['evaluation_frame_ranges']=[[round(c['actual_start_seconds']*16),round(c['actual_end_seconds']*16)] for c in windows]
        rows=list(iter_pair_metrics(pair,config,backend=backend))
        events=report._load_events(Path(pair['cache_events']))
        result=dict(prompt_id=key,high_motion=None,cases=[])
        if high:
            assert len(rows)==1401
            previous={**aggregate_frames(rows,group_by_window=False)[0],
                      **report.join_cache(pair,0,1401,'online_local_reconstruction',events=events)}
            result['high_motion']=dict(prompt_id=key,seed=p['seed'],previous_frame={m:previous[m] for m in METRICS},
                                       tempokv={m:tk_video[key][m] for m in METRICS})
        for case in windows:
            start,end=round(case['actual_start_seconds']*16),round(case['actual_end_seconds']*16)
            selected=[r for r in rows if start<=r['frame_index']<end]
            assert len(selected)==case['frames']
            previous={**report.aggregate_case_quality(selected),
                      **report.join_cache(pair,start,end,'online_local_reconstruction',events=events)}
            result['cases'].append(dict(case_id=case['case_id'],category=case['category'],prompt_id=key,
                actual_start_seconds=case['actual_start_seconds'],actual_end_seconds=case['actual_end_seconds'],
                previous_frame={m:previous[m] for m in METRICS},tempokv={m:case[m] for m in METRICS}))
        write_json(out/f'evaluation/prompt_{pid}.json',result)
        print(f'evaluated selected prompt {pid}; full_video={high}; hard_windows={len(windows)}',flush=True)


def main(args):
    base=Path(args.base).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    write_json(out/'status.json',dict(status='waiting',phase='wait_for_90_video_motion_groups',pid=os.getpid(),
        scope='high-motion full videos and difficult windows only'))
    analysis=base/args.analysis_subdir
    while not (analysis/'completion.json').is_file():
        state=read_json(base/'status.json')
        if state.get('status') in ('failed','group_counts_need_review'):raise RuntimeError(f'upstream motion90: {state}')
        time.sleep(30)
    videos=read_csv(analysis/'motion_videos.csv')
    cases=read_json(analysis/'difficult_cases.json')['cases']
    if args.case_category:cases=[c for c in cases if c['category'] in args.case_category]
    high=[] if args.difficult_only else sorted(int(r['prompt_id']) for r in videos if r['motion_group']=='high')
    assert args.difficult_only or len(high)==30
    difficult=sorted({int(c['prompt_id']) for c in cases})
    selected=sorted(set(high)|set(difficult))
    write_json(out/'selection.json',dict(high_motion_prompt_ids=high,difficult_prompt_ids=difficult,
        selected_prompt_ids=selected,case_ids=[c['case_id'] for c in cases],selection_basis='BF16-only high-motion groups and BF16-only hard-window selection',
        evaluation_scope='full high-motion videos; low/medium videos only within selected hard windows'))
    if not selected:
        write_json(out/'status.json',dict(status='no_eligible_difficult_cases',pid=os.getpid()));return
    config=read_json(base/'config.json')
    from tempokv.config import load_config
    codec=replace(load_config(config['codec_config']),k_predictor_mode='identity',v_predictor_mode='identity',
                  k_history_length=1,v_history_length=1)
    write_json(out/'identity_codec.json',asdict(codec))
    num_gpus=min(4,len(selected))
    config.update(codec_config=str(out/'identity_codec.json'),codec_overrides=asdict(codec),k_params=None,v_params=None,
        selected_prompt_ids=selected,start_index=0,end_index=90,num_gpus=num_gpus,skip_existing=True,method='previous_frame')
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='running',phase='selected_previous_frame_generation',n_prompts=len(selected),pid=os.getpid()))
    job=build_job(config,out/'runs/previous_frame',mode='tempokv',name='previous_frame_high_and_difficult')
    job.environment.update(CUDA_VISIBLE_DEVICES=','.join(map(str,range(num_gpus))),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    execute_jobs([job],resume=True)
    actual={int(p.stem.split('_')[0]) for p in (out/'runs/previous_frame/videos').glob('*.mp4')}
    assert actual==set(selected),'Generation must contain exactly the selected prompts'
    declared=read_json(out/'runs/previous_frame/videos/summary.json')
    generated=[r for shard in declared['rank_summaries'] for r in shard['videos']]
    expected={int(p['prompt_id']):p for p in read_json(base/args.pairs_file)['pairs']}
    assert {r['prompt_index'] for r in generated}==set(selected)
    assert all(r['noise_seed']==expected[r['prompt_index']]['seed'] and r['sample_index']==0 for r in generated)
    partition_events(out/'runs/previous_frame',selected,out/'cache_by_prompt')
    write_json(out/'status.json',dict(status='running',phase='selected_metrics',pid=os.getpid()))
    def worker(index):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'OMP_NUM_THREADS':'4'}
        with (out/f'evaluation_{index}.log').open('a') as log:
            subprocess.run([sys.executable,__file__,'--base',str(base),'--output',str(out),'--analysis-subdir',args.analysis_subdir,'--pairs-file',args.pairs_file,'--shard',str(index)],env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(worker,i) for i in range(4)]:future.result()
    evaluations=[read_json(out/f'evaluation/prompt_{pid}.json') for pid in selected]
    high_rows=[r['high_motion'] for r in evaluations if r['high_motion'] is not None]
    case_rows=[c for r in evaluations for c in r['cases']]
    assert len(high_rows)==len(high) and len(case_rows)==len(cases)
    write_json(out/'comparisons.json',dict(high_motion=high_rows,difficult_cases=case_rows))
    summary=[]
    for category,members in [('high_motion',high_rows),*[(c,[r for r in case_rows if r['category']==c]) for c in sorted({r['category'] for r in case_rows})]]:
        for method in ('previous_frame','tempokv'):
            entry=dict(category=category,method=method,n=len(members),n_source_videos=len({r['prompt_id'] for r in members}))
            for metric in METRICS:
                values=[r[method][metric] for r in members if r[method][metric] is not None]
                entry[metric]=sum(values)/len(values) if values else None
            summary.append(entry)
    write_csv(out/'comparison_summary.csv',summary)
    write_json(out/'status.json',dict(status='complete',n_high_motion=len(high),n_difficult_cases=len(case_rows),pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--analysis-subdir',default='analysis90');parser.add_argument('--case-category',nargs='+')
    parser.add_argument('--pairs-file',default='pairs90.json')
    parser.add_argument('--difficult-only',action='store_true')
    parser.add_argument('--shard',type=int);args=parser.parse_args()
    try:(evaluate_shard if args.shard is not None else main)(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
