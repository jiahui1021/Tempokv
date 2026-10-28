"""E5 prevents clean/reselection bypasses and reports complete actual access cost."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from tempokv.research.config import ResearchConfig
from tempokv.research.context import ProbeContext,FrozenAttention
from tempokv.research.memory_rollout import MemoryController
from tempokv.research.rollout import ReplayCase
from tempokv.research.archive import select_archive


def config(retention=0.,scope='selected',sink=0,recent=0):
    return ResearchConfig.from_dict({'experiments':['E5'],'selection':{'layers':[0],'heads':[0],'chunk_indices':[1],'denoise_step_indices':[0]},
        'sampling':{'protect_recent_latent_frames':recent,'queries_per_head':1},'sink_size':sink,
        'intervention':{'target_layer':0,'target_head':0,'target_chunk':1,'paired_rollout':True,'e5_scope':scope},
        'compression':{'retention':[retention]}})


def fixture(cfg,controller):
    state=SimpleNamespace(phase='denoise',chunk_id=1,step_index=0,active=True,config=cfg)
    calls=[]
    def attention(q,k,v):
        calls.append((state.phase,tuple(v[0,:,0,0].tolist())))
        weights=torch.einsum('bqhd,bkhd->bhqk',q.float(),k.float()).softmax(-1)
        return torch.einsum('bhqk,bkhd->bqhd',weights,v.float()).to(q.dtype)
    def freeze(layer,head,sample,q,k,v,metadata):
        start=metadata['current_start'];positions=metadata['absolute_positions']
        context=ProbeContext('test',0,0,layer,head,state.chunk_id,state.phase,state.step_index,None,start,1,1,1,
            torch.tensor([start]),torch.tensor(positions),cfg.sink_size,cfg.sampling.protect_recent_latent_frames)
        return FrozenAttention(context,q[:,:,head:head+1],k[:,:,head:head+1],v[:,:,head:head+1],attention),[]
    state.freeze_window=freeze
    def run(values,start,layer=0):
        heads=2 if cfg.intervention.e5_scope=='all' else 1
        q=torch.ones(1,1,heads,2,dtype=torch.bfloat16)
        k=torch.zeros(1,len(values),heads,2,dtype=q.dtype)
        v=torch.tensor(values,dtype=q.dtype).reshape(1,-1,1,1).expand(1,-1,heads,2).clone()
        baseline=attention(q,k,v)
        return controller.on_attention(state,layer,q,k,v,{'current_start':start,'absolute_positions':list(range(len(values)))},attention,baseline)
    return state,calls,run


def test_clean_refresh_cannot_read_removed_history_and_all_scope_enforces_other_heads():
    cfg=config(scope='all');controller=MemoryController(cfg)
    case=ReplayCase('recent',(),kind='archive',metadata={'policy':'recent_only','retention':0.})
    controller.setup_pass(case,None);state,calls,run=fixture(cfg,controller)
    assert controller.wants(state,7)
    assert torch.count_nonzero(run([9,6,0],2,layer=7))==0
    state.phase='clean_refresh';state.step_index=None
    assert controller.wants(state,7)
    assert torch.count_nonzero(run([9,6,0],2,layer=7))==0
    # Excluding the explicit diagnostic baseline, each head uses only current.
    assert calls[-2:]==[('clean_refresh',(0.,)),('clean_refresh',(0.,))]
    clean=[r for r in controller.local_rows if r['phase']=='clean_refresh']
    assert len(clean)==2 and all(r['actual_retained_tokens']==1 and r['archive_mask_reused'] for r in clean)
    assert all(r['kept_abs_positions']==[2] for r in clean)


def test_evicted_rows_do_not_reenter_later_scoring_or_attention(monkeypatch):
    import tempokv.research.memory_rollout as module
    cfg=config(.5);controller=MemoryController(cfg)
    controller.setup_pass(ReplayCase('attn',(),kind='archive',metadata={'policy':'attention_archive','retention':.5}),None)
    state,calls,run=fixture(cfg,controller);scored=[]
    def mass(q,k,chunk):
        # First step ties retain oldest absolute key; later a hypothetical
        # high score on an evicted row cannot be observed by this scorer.
        scored.append(k.shape[1]);return torch.ones(k.shape[1])
    monkeypatch.setattr(module,'token_mass',mass)
    run([10,20,30],2)
    assert controller.archives[0,0]['positions'].tolist()==[0,2]
    state.chunk_id=2
    run([10,20,30,40],3)
    assert scored==[3,3]  # second call has positions[0,2,3], not full[0,1,2,3]
    assert 1 not in controller.archives[0,0]['positions'].tolist()
    state.phase='clean_refresh';state.step_index=None
    run([10,999,30,40],3)
    assert len(scored)==2 and all(999 not in values for phase,values in calls[-1:])


def test_complete_budget_includes_clean_and_preserves_protected_sink_recent():
    cfg=config(0.,sink=1,recent=1);controller=MemoryController(cfg)
    case=ReplayCase('recent',(),kind='archive',metadata={'policy':'recent_only','retention':0.})
    controller.setup_pass(case,None);state,calls,run=fixture(cfg,controller)
    run([1,2,3,4],3)
    state.phase='clean_refresh';state.step_index=None;run([1,2,3,4],3)
    assert all(r['kept_abs_positions']==[0,2,3] for r in controller.local_rows)
    saved=[]
    probe=SimpleNamespace(prompt_index=0,seed=0,store=SimpleNamespace(write=lambda exp,row:saved.append(row)))
    controller.record_branch(probe,case,None,None,[])
    cost=next(r for r in saved if r.get('statistic')=='complete_archive_access_cost')
    assert cost['denoise_token_accesses']==3 and cost['clean_token_accesses']==3
    assert cost['total_token_accesses']==6


def test_full_window_and_replay_reset_are_exact():
    cfg=config(1.);controller=MemoryController(cfg)
    case=ReplayCase('recent',(),kind='archive',metadata={'policy':'recent_only','retention':1.})
    controller.setup_pass(case,None);state,calls,run=fixture(cfg,controller)
    actual=run([1,2,3],2);assert float(actual[0,0,0,0])==2
    controller.setup_pass(case,None);assert controller.archives=={}
    assert config(scope='selected').intervention.e5_scope=='selected'
    with pytest.raises(ValueError):config(scope='unknown')


def test_explicit_archive_budget_is_exact_and_validated():
    cfg=config();controller=MemoryController(cfg);controller.setup_pass(None,None)
    state,calls,run=fixture(cfg,controller)
    q=torch.ones(1,1,1,2);k=torch.zeros(1,5,1,2);v=torch.ones_like(k)
    window,_=state.freeze_window(0,0,0,q,k,v,{'current_start':4,'absolute_positions':list(range(5))})
    chosen=select_archive(window,policy='recent_only',retention=.2,token_budget=3)
    assert chosen['eligible_kept_tokens']==3
    for budget in (-1,5,True,1.5):
        with pytest.raises(ValueError):select_archive(window,policy='recent_only',retention=.2,token_budget=budget)
