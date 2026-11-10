"""Validate saved outputs after a final collective failure, then evaluate only.

The failed generation record and its outputs remain unchanged. Validation and
measurements are written to a new directory; no generation summary is invented.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256
from rebuttal.common.video_metrics import write_csv
from rebuttal.experiments.motion.previous_selected import METRICS,partition_events


def main(args):
    import av
    base=Path(args.base).resolve()
    source=Path(args.source).resolve()
    out=Path(args.output).resolve()
    if out.exists():
        raise FileExistsError('Use a fresh output directory for recovery')
    out.mkdir(parents=True)
    write_json(out/'status.json',dict(status='running',phase='validate_saved_outputs',pid=os.getpid()))
    selection=read_json(source/'selection.json')
    selected=selection['selected_prompt_ids']
    assert selected and not selection['high_motion_prompt_ids'], 'This recovery is difficult-windows only'
    config=read_json(source/'config.json')
    generation=source/'runs/previous_frame'
    record=read_json(generation/'rebuttal_generation.json')
    assert record['status']=='failed'
    log=(generation/'generation.log').read_text()
    assert 'OpType=ALLGATHER' in log and 'Watchdog caught collective operation timeout' in log
    expected={int(p['prompt_id']):p for p in read_json(base/'pairs.json')['pairs']}
    artifacts={str(Path(a['path']).resolve()):a['sha256'] for a in record['artifacts']}
    actual={int(p.stem.split('_')[0]) for p in (generation/'videos').glob('*.mp4')}
    assert actual==set(selected)
    evidence=[]
    for pid in selected:
        video=generation/f'videos/{pid}_0.mp4'
        checksum=file_sha256(video)
        assert artifacts[str(video)]==checksum
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            assert float(stream.average_rate)==16
            count=sum(1 for _ in container.decode(video=0))
        assert count==config['frames']==1401
        evidence.append(dict(prompt_id=pid,path=str(video),sha256=checksum,decoded_frames=count,
                             seed=expected[pid]['seed']))
    prompts=Path(config['prompts']).read_text().splitlines()
    import hashlib
    covered=defaultdict(set)
    for path in sorted(generation.glob('cache_events.rank-*.jsonl')):
        for line in path.open():
            event=json.loads(line)
            pid=int(event['prompt_id'])
            assert pid in selected
            assert event['seed']==expected[pid]['seed']
            assert event['sample_id']==0
            assert event['prompt_sha256']==hashlib.sha256(prompts[pid].strip().encode()).hexdigest()
            covered[(pid,event['role'],int(event['layer']))].add(int(event['absolute_latent_frame']))
    for pid in selected:
        for role in ('k','v'):
            layers={layer for p,r,layer in covered if p==pid and r==role}
            assert layers==set(range(30)),(pid,role,layers)
            for layer in layers:
                assert covered[(pid,role,layer)]==set(range(351)),(pid,role,layer)
    write_json(out/'recovery_validation.json',dict(status='complete',source=str(source),
        source_generation_status=record['status'],source_generation_record_sha256=file_sha256(generation/'rebuttal_generation.json'),
        source_failure='final NCCL all-gather timeout on empty fourth rank',
        scope='reuse fully decoded saved videos and complete cache events; do not rerun generation',
        cache_coverage='all351 latent frames,30layers,K and V per selected prompt',videos=evidence))
    write_json(out/'selection.json',selection)
    write_json(out/'config.json',config)
    (out/'runs').mkdir()
    (out/'runs/previous_frame').symlink_to(generation,target_is_directory=True)
    partition_events(generation,selected,out/'cache_by_prompt')
    write_json(out/'status.json',dict(status='running',phase='selected_metrics',n_prompts=len(selected),pid=os.getpid()))
    def worker(index):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        with (out/f'evaluation_{index}.log').open('a') as handle:
            subprocess.run([sys.executable,str(ROOT/'rebuttal/experiments/motion/previous_selected.py'),
                '--base',str(base),'--output',str(out),'--analysis-subdir','analysis','--pairs-file','pairs.json',
                '--shard',str(index)],env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=min(4,len(selected))) as pool:
        for future in [pool.submit(worker,i) for i in range(min(4,len(selected)))]:
            future.result()
    evaluations=[read_json(out/f'evaluation/prompt_{pid}.json') for pid in selected]
    cases=[c for row in evaluations for c in row['cases']]
    assert len(cases)==len(selection['case_ids'])
    assert {c['case_id'] for c in cases}==set(selection['case_ids'])
    write_json(out/'comparisons.json',dict(high_motion=[],difficult_cases=cases))
    summary=[]
    for category in sorted({c['category'] for c in cases}):
        members=[c for c in cases if c['category']==category]
        for method in ('previous_frame','tempokv'):
            entry=dict(category=category,method=method,n=len(members),n_source_videos=len({c['prompt_id'] for c in members}))
            for metric in METRICS:
                values=[c[method][metric] for c in members]
                assert all(v is not None for v in values),(category,method,metric)
                entry[metric]=sum(values)/len(values)
            summary.append(entry)
    write_csv(out/'comparison_summary.csv',summary)
    write_json(out/'status.json',dict(status='complete',n_high_motion=0,n_difficult_cases=len(cases),
        n_source_videos=len(selected),source=str(source),visual_validation='pending',pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',pid=os.getpid(),traceback=traceback.format_exc()))
        raise
