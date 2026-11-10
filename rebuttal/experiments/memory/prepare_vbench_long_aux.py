"""Warm official DINOv2 and DreamSim caches used by VBench-Long."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from rebuttal.common.manifest import write_json


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',dict(status='running',pid=os.getpid()))
    cache=Path.home()/'.cache'
    clips=cache/'clip';clips.mkdir(parents=True,exist_ok=True)
    for name in ('ViT-B-32.pt','ViT-L-14.pt'):
        target=clips/name
        source=Path(args.source)/'dependencies/vbench_cache/clip_model'/name
        if not target.exists():
            target.symlink_to(source)
    def dino():
        model=torch.hub.load('facebookresearch/dinov2','dinov2_vitb14',trust_repo=True).cpu().eval()
        assert sum(p.numel() for p in model.parameters())>1_000_000
        write_json(out/'dinov2.json',dict(status='complete',model='official dinov2_vitb14',scope='CPU model initialization'))
    def dream():
        from dreamsim import dreamsim
        model,_=dreamsim(pretrained=True,device='cpu',cache_dir=str(cache))
        assert sum(p.numel() for p in model.parameters())>1_000_000
        write_json(out/'dreamsim.json',dict(status='complete',model='official DreamSim default',scope='CPU model initialization'))
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,future in [('dinov2',pool.submit(dino)),('dreamsim',pool.submit(dream))]:
            try:
                future.result()
            except Exception:
                error=dict(model=name,status='failed',traceback=traceback.format_exc())
                write_json(out/f'{name}.json',error);errors.append(error)
    write_json(out/'status.json',dict(status='complete' if not errors else 'failed',errors=errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
