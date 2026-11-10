"""Populate missing official VBench assets in a staging directory first."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import write_json,file_sha256


def main(args):
    source=Path(args.source).resolve()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    cache=out/'cache';cache.mkdir()
    existing=source/'dependencies/vbench_cache'
    for path in existing.rglob('*'):
        if path.is_file():
            destination=cache/path.relative_to(existing)
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.link(path,destination)
    os.environ['VBENCH_CACHE_DIR']=str(cache)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    wgetrc=out/'wgetrc'
    wgetrc.write_text('timeout = 45\ntries = 2\n')
    os.environ['WGETRC']=str(wgetrc)
    os.environ['GIT_TERMINAL_PROMPT']='0'
    write_json(out/'status.json',dict(status='running',phase='official_asset_download',pid=os.getpid()))
    import torch
    from vbench.utils import init_submodules
    dimensions=['subject_consistency','background_consistency','motion_smoothness','dynamic_degree',
                'aesthetic_quality','imaging_quality','overall_consistency']
    init_submodules(dimensions,local=True)
    names=['dino_model/dino_vitbase16_pretrain.pth','clip_model/ViT-B-32.pt','clip_model/ViT-L-14.pt',
           'amt_model/amt-s.pth','raft_model/models/raft-things.pth',
           'aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth',
           'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth','ViCLIP/ViClip-InternVid-10M-FLT.pth']
    rows=[]
    for name in names:
        path=cache/name
        assert path.is_file() and path.stat().st_size>1000,name
        # CLIP .pt files are TorchScript archives; other weights are dictionaries.
        if name.startswith('clip_model/'):
            value=torch.jit.load(str(path),map_location='cpu')
        else:
            value=torch.load(path,map_location='cpu',weights_only=False)
            assert isinstance(value,dict),name
        del value
        rows.append(dict(path=name,size_bytes=path.stat().st_size,sha256=file_sha256(path)))
    assert (cache/'dino_model/facebookresearch_dino_main/hubconf.py').is_file()
    for path in cache.rglob('*'):
        if path.is_file():
            destination=existing/path.relative_to(cache)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():
                os.link(path,destination)
    write_json(out/'status.json',dict(status='complete',assets=rows,
        scope='official VBench assets deserialized on CPU; metric execution remains pending'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception:
        write_json(Path(args.output)/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise
