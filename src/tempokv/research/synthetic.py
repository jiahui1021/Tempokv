"""Explicit CPU protocol fixtures, not a video model or research evidence."""
from __future__ import annotations
from dataclasses import replace
import torch
from tempokv.spectral.sensitivity import reference_attention
from tempokv.spectral_kv.model_spec import ModelSpec
from tempokv.spectral_kv.rope3d import full_positions,rope_apply
from .probe import ResearchProbe


def run_synthetic(config):
    if config.intervention.paired_rollout:
        from .synthetic_rollout import run_synthetic_paired
        return run_synthetic_paired(config)
    if config.source!="bf16":raise ValueError("synthetic fixtures exercise BF16; no trained TempoKV parameters are fabricated")
    heads=max(config.selection.heads)+1;layers=config.selection.layers;dim=8;h,w=3,5;fs=h*w
    block=3;frames=config.latent_frames
    probe=ResearchProbe(config,model_metadata={"model":"deterministic_CPU_fixture","checkpoint":"none","grid":[h,w],
        "model_layers_evaluated":list(layers),"head_count":heads,"head_dim":dim,"synthetic_only":True})
    probe.begin(prompt_index="synthetic",origin="synthetic_cpu")
    generator=torch.Generator().manual_seed(config.seed)
    caches={layer:{} for layer in layers}
    steps=max(config.selection.denoise_step_indices)+1
    global_state=torch.get_rng_state().clone()
    try:
        for chunk,frame0 in enumerate(range(0,frames,block)):
            count=min(block,frames-frame0)
            if config.local_attn_size!=-1 and count+config.sink_size>config.local_attn_size:
                raise ValueError("synthetic window must contain current chunk and preserved sink")
            for step in range(steps+1):
                clean=step==steps
                probe.set_call(chunk_id=chunk,phase="clean_refresh" if clean else "denoise",step_index=None if clean else step,timestep=0 if clean else steps-step)
                for layer in layers:
                    q,k,v=[torch.randn(1,count*fs,heads,dim,generator=generator).to(torch.bfloat16) for _ in range(3)]
                    grid=[[count,h,w]];spec=ModelSpec(max(layers)+1,heads,dim,(h,w),count,(4,2,2))
                    probe.on_projected_native(layer,q_pre=q,k_pre=k,v=v,grid_sizes=grid,current_start=frame0*fs)
                    positions=full_positions(frame0,count,(h,w));q_post,k_post=[rope_apply(x,*positions,spec) for x in (q,k)]
                    for fi in range(count):caches[layer][frame0+fi]=(k_post[:,fi*fs:(fi+1)*fs],v[:,fi*fs:(fi+1)*fs])
                    all_frames=sorted(caches[layer])
                    if config.local_attn_size!=-1:
                        sink=[f for f in all_frames if f<config.sink_size]
                        recent=[f for f in all_frames if f>=config.sink_size][-(config.local_attn_size-len(sink)):]
                        active=sink+recent
                    else:active=all_frames
                    key,value=[torch.cat([caches[layer][f][j] for f in active],dim=1) for j in range(2)]
                    absolute=[f*fs+p for f in active for p in range(fs)]
                    offset=active.index(frame0)*fs
                    actual=probe.on_attention_native(layer,q_post=q_post,k_post=key,v=value,metadata={"grid_sizes":grid,"current_start":frame0*fs,
                        "local_start_index":offset,"attention_start":0,"local_end_index":key.shape[1],"frame_seqlen":fs,
                        "sink_frames":config.sink_size,"contiguous":False,"absolute_positions":absolute,"rope_precision":"float64",
                        "physical_kv_bytes":(key.numel()+value.numel())*key.element_size()},attention_fn=reference_attention)
                    expected=reference_attention(q_post,key,value)
                    if actual is not None:torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    for f in list(caches[layer]):
                        if f not in active:del caches[layer][f]
        if not torch.equal(global_state,torch.get_rng_state()):raise AssertionError("observer changed global RNG")
    except BaseException as exc:
        probe.finish(status="failed",extra={"error":str(exc)});raise
    return probe.finish(extra={"milestones_verified_by_fixture":["M0","M1"],"real_gpu_generation":"not_run"})
