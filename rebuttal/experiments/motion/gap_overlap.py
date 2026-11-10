"""Generate targeted supplements while Previous-frame metrics are evaluated."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json
from rebuttal.common.video_metrics import resolve_pairs
from rebuttal.adapters.generation import execute_jobs
from rebuttal.experiments.motion import report,previous_selected
from rebuttal.experiments.motion.previous_overlap import cpu_output_job
from rebuttal.experiments.motion.previous_selected import partition_events
from rebuttal.experiments.motion.expand90_finish import merge_quality_tables,case_tables,helper,cache_name
from rebuttal.experiments.motion.fill_gaps import PROMPTS
from rebuttal.run import execute


def main(args):
    parent=Path(args.parent).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    prompts=out/'targeted_prompts.txt'
    if not prompts.exists():prompts.write_text('\n'.join(PROMPTS)+'\n')
    write_json(out/'status.json',dict(status='waiting',phase='wait_previous_generation_only',pid=os.getpid()))
    while not (parent/'generation_complete.json').exists():
        if read_json(parent/'status.json').get('status')=='failed':raise RuntimeError('Previous-frame generation failed')
        time.sleep(30)
    source=parent.parent/'expanded90_20261109'
    config=read_json(source/'config.json')
    config.update(prompts=str(prompts),start_index=0,end_index=8,num_gpus=2,selected_prompt_ids=list(range(8)),
        skip_existing=True,output_dir=str(out))
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='running',phase='generation_overlapped_with_previous_metrics',pid=os.getpid()))
    def generate(mode,gpus):
        job=cpu_output_job(config,out/'runs'/mode,mode=mode,name=f'targeted_supplement_{mode}')
        job.environment.update(CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        return execute_jobs([job],resume=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(generate,'bf16','0,1'),pool.submit(generate,'tempokv','2,3')]:future.result()
    write_json(out/'generation_complete.json',dict(status='complete',scope='8 paired targeted videos only'))
    partition_events(out/'runs/tempokv',range(8),out/'cache_by_prompt')
    old_motion=source.parent
    cfg=read_json(old_motion/'manifest.json')['config']
    cfg.update(reference_dir=str(out/'runs/bf16/videos'),candidate_dir=str(out/'runs/tempokv/videos'),
        prompts=str(prompts),start_index=0,end_index=8,quality_output_dir=str(out/'quality'),output_dir=str(out/'analysis'))
    cfg.pop('pairs_manifest',None);cfg.pop('motion_groups',None)
    pairs=resolve_pairs(cfg)
    for pair in pairs:
        pair['comparison_id']='targeted_occlusion_scene_change_supplement'
        pair['cache_events']=str(out/f"cache_by_prompt/{pair['prompt_id']}.jsonl")
    write_json(out/'pairs.json',dict(pairs=pairs));write_json(out/'new_pairs.json',dict(pairs=pairs))
    cfg['pairs_manifest']=str(out/'pairs.json');write_json(out/'analysis_config.json',cfg)
    write_json(out/'status.json',dict(status='running',phase='supplement_analysis',pid=os.getpid()))
    def worker(index):
        directory=out/'shards'/str(index);directory.mkdir(parents=True,exist_ok=True)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'OMP_NUM_THREADS':'4'}
        with (directory/'analysis.log').open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'rebuttal/experiments/motion/expand90_finish.py'),
                '--source',str(source),'--output',str(out),'--shard',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(worker,i) for i in range(4)]:future.result()
    merge_quality_tables([out/'shards'/str(i)/'quality' for i in range(4)],out/'quality',cfg,8)
    original=helper(source)
    def cached(pair,settings,backend):
        path=out/'flow_cache'/cache_name(pair,original);entry=read_json(path)
        original.validate_cache_recipe(entry,original.recipe(pair,settings),path)
        return entry['flow_rows'],entry['motion_summary']
    report.analyze_reference=cached
    report.make_flow_backend=lambda settings:SimpleNamespace(metadata=read_json(old_motion/'flow_backend.json'))
    execute('motion',cfg,'evaluate',out/'analysis',resume=False);case_tables(out/'analysis')
    cases=read_json(out/'analysis/difficult_cases.json')['cases']
    counts={category:sum(c['category']==category for c in cases) for category in ('occlusion_disocclusion','abrupt_scene_change')}
    write_json(out/'analysis/completion.json',dict(status='complete',candidate_counts=counts,visual_validation='pending'))
    write_json(out/'status.json',dict(status='metrics_complete_visual_review_pending',candidate_counts=counts,pid=os.getpid()))
    previous_selected.build_job=cpu_output_job
    previous_selected.main(SimpleNamespace(base=str(out),output=str(out/'previous_frame'),analysis_subdir='analysis',
        pairs_file='pairs.json',difficult_only=True,case_category=['occlusion_disocclusion','abrupt_scene_change'],shard=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
