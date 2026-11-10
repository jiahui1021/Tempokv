from dataclasses import replace
import torch

from tempokv.config import CodecConfig
from tempokv.fitting import fit_predictors, _reconstruct_temporal_trace, _reconstruct_cross_kv_trace
from rebuttal.adapters.experimental_codec import ResearchTemporal, ResearchCross, encode_tensor
from rebuttal.experiments.paper_architecture.storage import WeightTensor, storage_bytes
from rebuttal.experiments.paper_architecture.temporal import build_temporal
from rebuttal.experiments.paper_architecture.cross import build_cross
from rebuttal.experiments.paper_architecture.history import build_history


def test_packed_weight_layer_selection_and_roundtrip():
    x=torch.linspace(-1,1,3*2*5*7).reshape(3,2,5,7)
    for precision in ['fp32','fp16','int8','int4']:
        w=WeightTensor.from_tensor(x,precision)
        torch.testing.assert_close(w.materialize(1),w.materialize()[1])
        torch.testing.assert_close(WeightTensor.from_state(w.state()).materialize(),w.materialize())
        if precision=='int4':assert w.codes.numel()==3*2*5*4


def test_paper_original_closed_loop_matches_production_helpers():
    torch.manual_seed(4)
    k=torch.randn(2,4,7,2,8);v=.4*k+torch.randn_like(k)*.2
    cfg=CodecConfig(k_bits=4,v_bits=4,anchor_bits=4,group_size=4,stride=1,
        k_history_length=2,v_history_length=1,k_predictor_mode='layer_head',
        cross_kv_weight_mode='layer_head',fit_mode='native',use_triton=False,cross_matmul_precision='ieee')
    kp,vp=fit_predictors(k,v,config=cfg,iterations=2,temporal_ridge=.1,cross_ridge=.1,
        v_history_ridge=0.,v_history_mode='diagonal',max_gain=1.,fit_history_source='reconstructed')
    rt=ResearchTemporal.from_production(kp,2,2);rc=ResearchCross.from_production(vp,'layer_head',2,2)
    tm=build_temporal(k,rt,cfg,{'kind':'paper'})[0]
    cm=build_cross(k,v,rc,cfg,{'kind':'paper'})[0]
    hm=build_history(k,v,cm,cfg,{'kind':'paper'})[0]
    expected_k=_reconstruct_temporal_trace(k,kp,cfg,bits=4)
    expected_v=_reconstruct_cross_kv_trace(expected_k,v,vp,cfg)
    for layer in range(2):
        _,kr=encode_tensor(k[layer].permute(0,2,1,3),tm,cfg,bits=4,layer=layer)
        _,vr=encode_tensor(v[layer].permute(0,2,1,3),cm,cfg,bits=4,layer=layer,source=kr)
        torch.testing.assert_close(kr.permute(0,2,1,3).float(),expected_k[layer].float(),atol=0,rtol=0)
        torch.testing.assert_close(vr.permute(0,2,1,3).float(),expected_v[layer].float(),atol=0,rtol=0)
        _,hr=encode_tensor(v[layer].permute(0,2,1,3),hm,cfg,bits=4,layer=layer,source=kr)
        torch.testing.assert_close(hr,vr,atol=0,rtol=0)


def test_paper_storage_budget_formulas_are_actual_payloads():
    layers,heads,dim,p=30,12,128,2
    baseline=torch.empty(layers,heads,p,p,dim)
    blocks=torch.empty(layers,heads,p,p,dim//2,2,2,dtype=torch.float16)
    a=torch.empty(layers,heads,p,p,dim,1,dtype=torch.float16)
    b=torch.empty(layers,heads,p,p,1,dim,dtype=torch.float16)
    bias=torch.empty(layers,heads,p,dim)
    assert storage_bytes([baseline,bias])==storage_bytes([blocks,bias])==storage_bytes([a,b,bias])==1105920
    w=torch.zeros(layers,heads,dim,dim)
    assert storage_bytes(WeightTensor.from_tensor(w,'int8').tensors())==5990400
