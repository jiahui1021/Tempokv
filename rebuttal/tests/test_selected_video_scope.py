import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rebuttal.common.video_metrics import iter_pair_metrics


def test_native_prompt_subset_keeps_original_indices_and_order():
    path=Path(__file__).resolve().parents[2]/'experiments/Self-Forcing/inference.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.prompt_indices is not None')
    env={'args':SimpleNamespace(prompt_indices='[89, 50, 73]',start_index=0),
         'end_index':90,'selected_indices':list(range(90)),'json':json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<subset>','exec'),env)
    assert env['selected_indices']==[50,73,89]
    assert [i*1_000_003 for i in env['selected_indices']]==[50_000_150,73_000_219,89_000_267]


def test_window_only_metrics_still_validate_entire_video():
    class Reader:
        fps=16;width=2;height=2;declared_frames=4;timestamp_policy='cfr'
        def __init__(self,path,decoder):self.path=path
        def __iter__(self):
            for index in range(4):
                yield SimpleNamespace(image=np.ones((2,2,3),dtype=np.uint8),timestamp=index/16)
        def close(self):pass
    calls=[]
    backend=SimpleNamespace(ssim=lambda a,b:calls.append(1) or 1.0)
    pair=dict(reference='reference.mp4',candidate='candidate.mp4',evaluation_frame_ranges=[[1,2],[3,4]])
    cfg=dict(metrics=['ssim'],expected_frames=4)
    rows=list(iter_pair_metrics(pair,cfg,backend=backend,reader_factory=Reader))
    assert [row['frame_index'] for row in rows]==[1,3] and len(calls)==2
    with pytest.raises(ValueError,match='expected 5 frames'):
        list(iter_pair_metrics(pair,{**cfg,'expected_frames':5},backend=backend,reader_factory=Reader))
