"""Start selected Previous-frame generation from BF16-only motion evidence."""
import argparse
from dataclasses import asdict,replace
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256
from rebuttal.common.video_metrics import read_csv
from rebuttal.adapters.generation import build_job,execute_jobs
from rebuttal.experiments.motion import report,previous_selected
from rebuttal.experiments.motion.groups import assign_groups
from rebuttal.experiments.motion.expand90_finish import helper,cache_name
from rebuttal.experiments.motion.recover_cached import analysis_ast


def cpu_output_job(config,folder,**kwargs):
    job=build_job(config,folder,**kwargs)
    original=str(ROOT/'rebuttal/adapters/worker.py')
    job.command=[str(ROOT/'rebuttal/adapters/overlap_worker.py') if part==original else part for part in job.command]
    job.environment['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    return job


def select_from_flow(base,out):
    config=read_json(base/'analysis_config.json');pairs=read_json(base/'pairs90.json')['pairs']
    assert len(pairs)==90
    original=helper(base);old_motion=base.parent
    current=Path(report.__file__).resolve();backup=current.with_name('report.py.before_indexing_20261109')
    assert analysis_ast(current)==analysis_ast(backup)
    videos=[];flows=[]
    for pair in pairs:
        old=int(pair['prompt_id'])<50
        path=(old_motion/'flow_cache' if old else base/'flow_cache')/cache_name(pair,original)
        entry=read_json(path);recipe=original.recipe(pair,config)
        if old:recipe['source_sha256'][str(current.relative_to(ROOT))]=file_sha256(backup)
        original.validate_cache_recipe(entry,recipe,path)
        videos.append({**entry['motion_summary'],**{k:pair[k] for k in ('model','prompt_id','seed','sample_id','method','comparison_id')}})
        flows.extend(entry['flow_rows'])
    directory=out/'bf16_selection';directory.mkdir(parents=True,exist_ok=True)
    groups=assign_groups(videos)
    counts={g:sum(r['motion_group']==g for r in videos) for g in ('low','medium','high')}
    assert counts=={'low':30,'medium':30,'high':30}
    cases=report.select_candidates(flows,pairs,config,directory)
    high=sorted(int(r['prompt_id']) for r in videos if r['motion_group']=='high')
    difficult=sorted({int(c['prompt_id']) for c in cases});selected=sorted(set(high)|set(difficult))
    selection=dict(high_motion_prompt_ids=high,difficult_prompt_ids=difficult,selected_prompt_ids=selected,
        case_ids=[c['case_id'] for c in cases],case_prompt_pairs=[[c['case_id'],str(c['prompt_id'])] for c in cases],
        selection_basis='all90 verified BF16 caches only; independent of candidate quality',
        evaluation_scope='full high-motion videos; low/medium only selected difficult windows')
    write_json(directory/'motion_groups.json',groups);write_json(out/'selection.json',selection)
    write_json(directory/'selection_ready.json',dict(status='ready',scope='BF16 sample selection only; not evaluation completion',
        group_counts=counts,n_selected=len(selected),n_difficult_candidates=len(cases)))
    return selection


def main(args):
    base=Path(args.base).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    write_json(out/'status.json',dict(status='running',phase='BF16_only_selection',pid=os.getpid()))
    selection=select_from_flow(base,out)
    config=read_json(base/'config.json')
    from tempokv.config import load_config
    codec=replace(load_config(config['codec_config']),k_predictor_mode='identity',v_predictor_mode='identity',k_history_length=1,v_history_length=1)
    write_json(out/'identity_codec.json',asdict(codec))
    config.update(codec_config=str(out/'identity_codec.json'),codec_overrides=asdict(codec),k_params=None,v_params=None,
        selected_prompt_ids=selection['selected_prompt_ids'],start_index=0,end_index=90,num_gpus=4,skip_existing=True,method='previous_frame')
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(status='running',phase='generation_overlapped_with_quality',pid=os.getpid(),
        n_high=30,n_selected=len(selection['selected_prompt_ids'])))
    job=cpu_output_job(config,out/'runs/previous_frame',mode='tempokv',name='previous_frame_high_and_difficult')
    job.environment.update(CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    execute_jobs([job],resume=True)
    write_json(out/'generation_complete.json',dict(status='complete',scope='generation only',selected_prompt_ids=selection['selected_prompt_ids']))
    write_json(out/'status.json',dict(status='waiting',phase='generation_done_waiting_for_reference_quality',pid=os.getpid()))
    while not (base/'analysis90/completion.json').exists():
        if read_json(base/'status.json').get('status')=='failed':raise RuntimeError('main90 analysis failed')
        time.sleep(30)
    videos=read_csv(base/'analysis90/motion_videos.csv');cases=read_json(base/'analysis90/difficult_cases.json')['cases']
    assert sorted(int(r['prompt_id']) for r in videos if r['motion_group']=='high')==selection['high_motion_prompt_ids']
    assert [[c['case_id'],str(c['prompt_id'])] for c in cases]==selection['case_prompt_pairs']
    previous_selected.build_job=cpu_output_job
    previous_selected.main(SimpleNamespace(base=str(base),output=str(out),analysis_subdir='analysis90',pairs_file='pairs90.json',
        difficult_only=False,case_category=None,shard=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()));raise
