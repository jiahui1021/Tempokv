"""Queued targeted occlusion/cut supplement; kept separate from main90."""
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
from rebuttal.adapters.generation import build_job,execute_jobs
from rebuttal.experiments.motion import report
from rebuttal.experiments.motion.previous_selected import partition_events
from rebuttal.experiments.motion.expand90_finish import merge_quality_tables,case_tables,helper,cache_name
from rebuttal.run import execute

PROMPTS=[
 'A locked-off camera watches a person in a red coat walk across a courtyard, disappear completely behind a broad concrete pillar for several seconds, then reappear on the other side. A continuous uncut shot.',
 'A stationary street camera watches a pedestrian waiting on the far sidewalk. A large city bus crosses the foreground and completely hides the pedestrian, then drives away to reveal the same pedestrian again. A continuous uncut shot.',
 'A fixed camera watches a woman standing inside a doorway. A solid wooden door slowly closes until she is fully hidden, then opens to reveal her again. A continuous uncut shot.',
 'A fixed camera watches a cyclist in the background. A large delivery truck passes close across the entire foreground, completely occluding the cyclist, then clears the view and reveals the cyclist again. A continuous uncut shot.',
 'A multi-shot film. First a static wide shot of a bright snowy mountain. At five seconds an instantaneous hard cut reveals a completely different dark indoor nightclub. The transition is a direct edit, with no dissolve, morphing, or camera movement.',
 'A multi-shot film. First a static close-up of a yellow flower in sunlight. At five seconds there is an abrupt hard cut to a wide blue underwater coral reef, a completely different location. No dissolve or morphing.',
 'A multi-shot film alternates distinct scenes with instantaneous hard cuts: a bright sandy desert, then a dark rainy neon city at night, then a snowy forest in daylight. Each shot lasts several seconds. No smooth transitions or morphing.',
 'A multi-shot film begins with a quiet sunlit kitchen. After five seconds, a sudden hard cut switches to a completely different shot of a moonlit ocean with a sailing ship. The edit is instantaneous; no pan, zoom, dissolve, or morphing.'
]


def main(args):
    parent=Path(args.parent).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    prompts=out/'targeted_prompts.txt'
    if not prompts.exists():prompts.write_text('\n'.join(PROMPTS)+'\n')
    write_json(out/'cohort.json',dict(name='targeted_occlusion_scene_change_supplement',n_prompts=8,
        occlusion_prompt_ids=list(range(4)),scene_change_prompt_ids=list(range(4,8)),
        interpretation='requested scene cuts are not treated as observed cuts; measured detection and visual verification required',
        excluded_from_main90_motion_groups=True))
    write_json(out/'status.json',dict(status='waiting',phase='after_high_and_difficult_previous_frame',pid=os.getpid()))
    while True:
        state=read_json(parent/'status.json')
        if state['status']=='complete':break
        if state['status']=='failed':raise RuntimeError('Previous-frame main follow-up failed')
        time.sleep(30)
    source=parent.parent/'expanded90_20261109'
    config=read_json(source/'config.json')
    config.update(prompts=str(prompts),start_index=0,end_index=8,num_gpus=2,selected_prompt_ids=list(range(8)),
        skip_existing=True,output_dir=str(out))
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='running',phase='targeted_paired_generation',pid=os.getpid()))
    def generate(mode,gpus):
        job=build_job(config,out/'runs'/mode,mode=mode,name=f'targeted_supplement_{mode}')
        job.environment.update(CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        return execute_jobs([job],resume=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(generate,'bf16','0,1'),pool.submit(generate,'tempokv','2,3')]:future.result()
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
    execute('motion',cfg,'evaluate',out/'analysis',resume=False)
    case_tables(out/'analysis')
    cases=read_json(out/'analysis/difficult_cases.json')['cases']
    counts={category:sum(c['category']==category for c in cases) for category in ('occlusion_disocclusion','abrupt_scene_change')}
    write_json(out/'analysis/completion.json',dict(status='complete',candidate_counts=counts,visual_validation='pending'))
    write_json(out/'status.json',dict(status='metrics_complete_visual_review_pending',candidate_counts=counts,pid=os.getpid()))
    subprocess.run([sys.executable,str(ROOT/'rebuttal/experiments/motion/previous_selected.py'),
        '--base',str(out),'--output',str(out/'previous_frame'),'--analysis-subdir','analysis','--pairs-file','pairs.json',
        '--difficult-only','--case-category','occlusion_disocclusion','abrupt_scene_change'],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
