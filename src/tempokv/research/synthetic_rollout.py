"""CPU AR denoising surrogate for validating paired hooks, not video evidence."""
from __future__ import annotations
import torch
from tempokv.spectral.sensitivity import reference_attention
from tempokv.spectral_kv.model_spec import ModelSpec
from tempokv.spectral_kv.rope3d import full_positions,rope_apply
from .interaction_rollout import make_controller
from .probe import ResearchProbe
from .rollout import RolloutOutput,paired_replay


def run_synthetic_paired(config):
    heads=max(config.selection.heads)+1;dim=8;h,w=2,3;fs=h*w
    steps=max(config.selection.denoise_step_indices)+1
    if config.intervention.target_step>=steps:raise ValueError("target step outside synthetic schedule")
    if config.intervention.target_layer not in config.selection.layers or config.intervention.target_head not in config.selection.heads:
        raise ValueError("synthetic paired target must be in selected layers/heads")
    torch.manual_seed(config.seed)
    initial=torch.randn(1,config.latent_frames,h,w,heads,dim)
    probe=ResearchProbe(config,model_metadata={"model":"CPU_AR_denoising_surrogate","checkpoint":"none","window":config.local_attn_size,
                        "synthetic_only":True,"chunk_frames":1,"denoise_steps":steps})
    controller=make_controller(config);probe.counterfactual=controller;roots=[]
    def run(case,recorder):
        if probe.active:probe.finish(extra={"case_id":controller.case.case_id if controller.case else "reference"})
        roots.append(probe.begin(prompt_index="synthetic_pair",origin="synthetic_reference" if case is None else "synthetic_counterfactual",
                                 metadata={"case_id":case.case_id if case else None}))
        controller.setup_pass(case,recorder)
        caches={layer:{} for layer in config.selection.layers};clean_outputs=[]
        def forward(noisy,chunk,step,phase):
            probe.set_call(chunk_id=chunk,phase=phase,step_index=step,timestep=0 if phase=="clean_refresh" or step==steps-1 else 900)
            recorder.record_rng("before_forward",(chunk,phase,step))
            current=noisy
            for layer in config.selection.layers:
                q=torch.tanh(current).reshape(1,fs,heads,dim).to(torch.bfloat16)
                k=(current*.9).reshape(1,fs,heads,dim).to(torch.bfloat16)
                v=(current*.8).reshape(1,fs,heads,dim).to(torch.bfloat16)
                grid=[[1,h,w]];spec=ModelSpec(max(config.selection.layers)+1,heads,dim,(h,w),1,(4,2,2))
                probe.on_projected_native(layer,q_pre=q,k_pre=k,v=v,grid_sizes=grid,current_start=chunk*fs)
                pos=full_positions(chunk,1,(h,w));q_post,k_post=[rope_apply(x,*pos,spec) for x in (q,k)]
                caches[layer][chunk]=(k_post,v)
                frames=sorted(caches[layer])
                if config.local_attn_size!=-1:
                    sink=[f for f in frames if f<config.sink_size];recent=[f for f in frames if f>=config.sink_size][-(config.local_attn_size-len(sink)):]
                    frames=sink+recent
                kk,vv=[torch.cat([caches[layer][f][role] for f in frames],dim=1) for role in (0,1)]
                offset=frames.index(chunk)*fs
                actual=probe.on_attention_native(layer,q_post=q_post,k_post=kk,v=vv,metadata={"grid_sizes":grid,"current_start":chunk*fs,
                    "local_start_index":offset,"attention_start":0,"frame_seqlen":fs,"sink_frames":config.sink_size,"contiguous":False,
                    "absolute_positions":[f*fs+i for f in frames for i in range(fs)],"rope_precision":"float64",
                    "physical_kv_bytes":(kk.numel()+vv.numel())*kk.element_size()},attention_fn=reference_attention)
                if actual is None:actual=reference_attention(q_post,kk,vv)
                current=.55*current+.25*actual.float().reshape_as(current)
                for f in list(caches[layer]):
                    if f not in frames:del caches[layer][f]
            if phase=="denoise":recorder.record_output(chunk,step,current);recorder.record_chunk(chunk,chunk,1)
            recorder.record_rng("after_forward",(chunk,phase,step))
            return current
        for chunk in range(config.latent_frames):
            # The branch's own finalized history influences the next chunk input.
            previous=clean_outputs[-1].mean() if clean_outputs else 0.
            noisy=initial[:,chunk:chunk+1].clone()+.15*previous
            for step in range(steps):
                denoised=forward(noisy,chunk,step,"denoise")
                if step<steps-1:noisy=denoised+.05*torch.randn_like(denoised)
            clean_outputs.append(denoised)
            forward(denoised,chunk,None,"clean_refresh")  # Only its cache writes are used.
        return RolloutOutput(torch.cat(clean_outputs,dim=1))
    def on_branch(case,trace,output,rows):controller.record_branch(probe,case,trace,output,rows)
    try:
        result=paired_replay(run,controller.cases,max_capture_mb=config.sampling.max_capture_mb,
            horizons=config.intervention.horizon_chunks,target_chunk=config.intervention.target_chunk,
            record_steps={(config.intervention.target_chunk,s) for s in range(steps)},on_branch=on_branch)
        controller.finalize(probe)
        probe.finish(extra={"milestones_verified_by_fixture":["M0","M1","M2"],"paired_branches":result.branch_summaries,
            "unrun_branch_ids":controller.unrun,"real_gpu_generation":"not_run"})
    except BaseException as exc:
        probe.finish(status="failed",extra={"error":str(exc)});raise
    finally:probe.counterfactual=None
    return roots[-1]
