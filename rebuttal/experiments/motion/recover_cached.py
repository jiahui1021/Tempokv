"""Recover aggregation in a new run directory using verified existing flow."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import read_json,write_json,file_sha256
from rebuttal.run import execute
from rebuttal.experiments.motion import report


def analysis_ast(path):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='analyze_reference')
    return ast.dump(node,include_attributes=False)


def main(args):
    runner=Path(args.runner).resolve();sys.path.insert(0,str(runner))
    import analysis_background as original
    source=Path(args.source).resolve();output=Path(args.output).resolve()
    current=Path(report.__file__).resolve()
    backup=current.with_name('report.py.before_indexing_20261109')
    # Aggregation changed; the actual optical-flow/camera computation must not.
    assert analysis_ast(current)==analysis_ast(backup),'Flow analysis function changed'
    config=dict(read_json(source/'manifest.json')['config'])
    config['output_dir']=str(output)
    backend_metadata=read_json(source/'flow_backend.json')
    counters={'flow_videos_reused':0}
    def cached(pair,cfg,backend):
        path=original.cache_path(pair)
        payload=read_json(path)
        expected=original.recipe(pair,cfg)
        expected['source_sha256'][str(current.relative_to(ROOT))]=file_sha256(backup)
        original.validate_cache_recipe(payload,expected,path)
        counters['flow_videos_reused']+=1
        print(f"verified flow cache {counters['flow_videos_reused']}: {pair['prompt_id']}",flush=True)
        return payload['flow_rows'],payload['motion_summary']
    report.analyze_reference=cached
    report.make_flow_backend=lambda settings:SimpleNamespace(metadata=backend_metadata)
    write_json(output/'recovery_provenance.json',dict(source=str(source),
        previous_report_sha256=file_sha256(backup),current_report_sha256=file_sha256(current),
        unchanged_flow_function_sha256=hashlib.sha256(analysis_ast(current).encode()).hexdigest(),
        scope='aggregation_only; original reference/settings/checkpoint/flow/camera recipe validated'))
    result=execute('motion',config,'evaluate',output,resume=False)
    assert int(result['n_video_pairs'])==50
    write_json(output/'completion.json',dict(status='complete',n_video_pairs=50,**counters))
    print(json.dumps(dict(status='complete',n_video_pairs=50,**counters)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ('runner','source','output'):parser.add_argument('--'+field,required=True)
    main(parser.parse_args())
