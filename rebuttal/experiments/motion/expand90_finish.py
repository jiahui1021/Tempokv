"""Finish the90-video motion cohort using existing50 and newly generated40."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256
from rebuttal.common.video_metrics import resolve_pairs,read_csv,write_csv
from rebuttal.experiments import temporal_window
from rebuttal.experiments.motion import report
from rebuttal.experiments.motion.recover_cached import analysis_ast
from rebuttal.run import execute


def helper(source):
    runner=ROOT.parent
    sys.path.insert(0,str(runner))
    import analysis_background
    return analysis_background


def cache_name(pair,original):
    return original.cache_path(pair).name


def collect_events(out):
    target=out/'online_cache_events.jsonl'
    rows={}
    for path in sorted((out/'runs/tempokv').glob('cache_events.rank-*.jsonl')):
        with path.open() as handle:
            for line in handle:
                event=json.loads(line)
                assert event['metric_scope']=='online_local_reconstruction'
                key=tuple(str(event[k]) for k in ('model','prompt_id','seed','sample_id','layer','role','absolute_latent_frame'))
                rows[key]=line
    assert {int(k[1]) for k in rows}==set(range(50,90))
    with target.open('w') as handle:
        for line in rows.values():handle.write(line)
    write_json(out/'cache_event_provenance.json',dict(files=[str(p) for p in sorted((out/'runs/tempokv').glob('cache_events.rank-*.jsonl'))],
        unique_events=len(rows),policy='latest event per complete prompt/seed/sample/layer/role/frame; handles partial retry prefixes'))
    return target


def shard(args):
    out=Path(args.output);source=Path(args.source);original=helper(source)
    config=read_json(out/'analysis_config.json')
    pairs=read_json(out/'new_pairs.json')['pairs'][args.shard::4]
    backend=report.make_flow_backend(config['flow'])
    for pair in pairs:
        path=out/'flow_cache'/cache_name(pair,original)
        recipe=original.recipe(pair,config)
        if path.exists():
            original.validate_cache_recipe(read_json(path),recipe,path)
        else:
            rows,summary=report.analyze_reference(pair,config,backend)
            write_json(path,dict(recipe=recipe,flow_rows=rows,motion_summary=summary))
        print(f"flow complete prompt {pair['prompt_id']}",flush=True)
    quality=out/'shards'/str(args.shard)/'quality'
    qconfig={**config,'pairs':pairs,'plots':False}
    qconfig.pop('pairs_manifest',None)
    if (quality/temporal_window.QUALITY_COMPLETION_FILE).exists():
        report.validate_quality_reuse(quality,pairs,qconfig)
    else:
        temporal_window.run(qconfig,'evaluate',quality)
    print(f'shard {args.shard} complete',flush=True)


def merge_quality(source,out,config,old_pairs,new_pairs):
    report.validate_quality_reuse(source,old_pairs,config)
    directories=[source,*[out/'shards'/str(i)/'quality' for i in range(4)]]
    merge_quality_tables(directories,out/'quality90',config,90)


def merge_quality_tables(directories,destination,config,expected_videos):
    destination.mkdir(parents=True,exist_ok=True)
    all_pairs=[]
    for directory in directories:
        declaration=read_json(directory/'video_pairs.json')['pairs']
        report.validate_quality_reuse(directory,declaration,config)
        all_pairs.extend(declaration)
    assert len(all_pairs)==expected_videos and len({p['prompt_id'] for p in all_pairs})==expected_videos
    write_json(destination/'video_pairs.json',dict(pairs=all_pairs,config=config))
    for filename in ('frames.csv','windows.csv','videos.csv'):
        rows=[]
        for directory in directories:rows.extend(read_csv(directory/filename))
        if filename=='frames.csv':assert len(rows)==expected_videos*1401
        if filename=='videos.csv':assert len(rows)==expected_videos
        write_csv(destination/filename,rows)
    write_json(destination/temporal_window.QUALITY_COMPLETION_FILE,dict(
        schema_version=temporal_window.QUALITY_COMPLETION_SCHEMA,status='complete',
        outputs=temporal_window.quality_output_fingerprints(destination)))
    write_json(destination/'reuse_provenance.json',dict(source_directories=[str(p) for p in directories],
        source_completion_sha256=[file_sha256(p/temporal_window.QUALITY_COMPLETION_FILE) for p in directories]))


def case_tables(directory):
    cases=read_json(directory/'difficult_cases.json')['cases']
    fields=('psnr_db','ssim','lpips','k_reconstruction_nmse','v_reconstruction_nmse')
    write_csv(directory/'difficult_case_metrics.csv',[{k:row.get(k) for k in (
        'case_id','category','prompt_id','seed','actual_start_seconds','actual_end_seconds','frames',*fields,
        'cache_missing_reason','manual_verified')} for row in cases])
    rows=[]
    for category in ('large_local_motion','large_camera_motion','occlusion_disocclusion','abrupt_scene_change'):
        members=[r for r in cases if r['category']==category]
        item=dict(category=category,n_clips=len(members),n_source_videos=len({r['prompt_id'] for r in members}),aggregation='equal_clip_mean')
        for field in fields:
            values=[r[field] for r in members if r.get(field) is not None]
            item[field]=sum(values)/len(values) if values else None
        rows.append(item)
    write_csv(directory/'difficult_category_metrics.csv',rows)
    return rows


def finish(args):
    source=Path(args.source).resolve();out=Path(args.output).resolve()
    original=helper(source)
    old_motion=source.parent/'motion_robustness'
    config=dict(read_json(old_motion/'manifest.json')['config'])
    events=collect_events(out)
    new_config={**config,'start_index':50,'end_index':90,
        'reference_dir':str(out/'runs/bf16/videos'),'candidate_dir':str(out/'runs/tempokv/videos')}
    new_config.pop('pairs_manifest',None);new_config.pop('pairs',None)
    new_pairs=resolve_pairs(new_config)
    assert len(new_pairs)==40 and {int(p['prompt_id']) for p in new_pairs}==set(range(50,90))
    for pair in new_pairs:pair['cache_events']=str(events)
    old_pairs=read_json(old_motion/'pairs.json')['pairs']
    assert len(old_pairs)==50
    pairs=old_pairs+new_pairs
    write_json(out/'new_pairs.json',dict(pairs=new_pairs))
    write_json(out/'pairs90.json',dict(pairs=pairs))
    config.update(start_index=0,end_index=90,pairs_manifest=str(out/'pairs90.json'),
        quality_output_dir=str(out/'quality90'),output_dir=str(out/'analysis90'))
    config.pop('pairs',None);config.pop('motion_groups',None)
    write_json(out/'analysis_config.json',config)
    def worker(index):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        path=out/'shards'/str(index);path.mkdir(parents=True,exist_ok=True)
        with (path/'analysis.log').open('a') as log:
            subprocess.run([sys.executable,__file__,'--source',str(source),'--output',str(out),'--shard',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in [pool.submit(worker,i) for i in range(4)]:future.result()
    merge_quality(source,out,config,old_pairs,new_pairs)
    current=Path(report.__file__).resolve();backup=current.with_name('report.py.before_indexing_20261109')
    assert analysis_ast(current)==analysis_ast(backup)
    def cached(pair,cfg,backend):
        old=int(pair['prompt_id'])<50
        path=(old_motion/'flow_cache' if old else out/'flow_cache')/cache_name(pair,original)
        payload=read_json(path);recipe=original.recipe(pair,cfg)
        if old:recipe['source_sha256'][str(current.relative_to(ROOT))]=file_sha256(backup)
        original.validate_cache_recipe(payload,recipe,path)
        return payload['flow_rows'],payload['motion_summary']
    from types import SimpleNamespace
    report.analyze_reference=cached
    report.make_flow_backend=lambda settings:SimpleNamespace(metadata=read_json(old_motion/'flow_backend.json'))
    summary=execute('motion',config,'evaluate',out/'analysis90',resume=False)
    counts={g['motion_group']:g['n_videos'] for g in summary['groups']}
    case_tables(out/'analysis90')
    complete=summary['n_video_pairs']==90 and counts=={'low':30,'medium':30,'high':30}
    write_json(out/'status.json',dict(status='complete' if complete else 'group_counts_need_review',
        n_video_pairs=summary['n_video_pairs'],group_counts=counts,pid=os.getpid()))
    if not complete:raise RuntimeError(f'Expected30/30/30 without splitting ties: {counts}')
    write_json(out/'analysis90/completion.json',dict(status='complete',n_video_pairs=90,group_counts=counts))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--shard',type=int)
    args=parser.parse_args()
    (shard if args.shard is not None else finish)(args)
