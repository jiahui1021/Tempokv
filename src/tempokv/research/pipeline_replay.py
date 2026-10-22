"""Paired inference adapter: replay original pipeline, not reference cache states."""
from __future__ import annotations
import torch
from .interaction_rollout import make_controller
from .rollout import RolloutOutput,paired_replay


def run_paired_pipeline(pipeline,base_inference,args,kwargs,config):
    probe=pipeline.research_probe
    noise=kwargs.get("noise",args[0] if args else None)
    if noise is None or noise.shape[0]!=1:raise ValueError("paired replay requires one sample per prompt/seed")
    initial=noise.detach().clone()
    desired=kwargs.get("return_latents",args[3] if len(args)>3 else False)
    controller=make_controller(config);probe.counterfactual=controller
    context=dict(pipeline._research_context)
    def run(case,recorder):
        if probe.active:probe.finish(extra={"replay_case":controller.case.case_id if controller.case else "reference"})
        pipeline._research_clean=False;pipeline._research_counts.clear();pipeline._research_starts.clear()
        origin="runtime_"+config.source+("_reference" if case is None else "_counterfactual")
        probe.begin(origin=origin,metadata={"paired_phase":"reference" if case is None else "branch",
                                           "case_id":case.case_id if case else None},**context)
        controller.setup_pass(case,recorder)
        call_args=list(args);call_kwargs=dict(kwargs)
        if call_args:call_args[0]=initial.clone();call_kwargs.pop("noise",None)
        else:call_kwargs["noise"]=initial.clone()
        if len(call_args)>3:call_args[3]=True;call_kwargs.pop("return_latents",None)
        else:call_kwargs["return_latents"]=True
        call_kwargs["decode_output"]=config.output.save_videos
        value=base_inference(*call_args,**call_kwargs)
        if not isinstance(value,tuple) or len(value)<2 or not isinstance(value[1],torch.Tensor):raise ValueError("paired pipeline requires explicit final clean latents")
        return RolloutOutput(value[1],value)
    def on_branch(case,trace,output,rows):controller.record_branch(probe,case,trace,output,rows)
    count=len(pipeline.denoising_step_list);target=config.intervention.target_chunk
    try:
        result=paired_replay(run,controller.cases,max_capture_mb=config.sampling.max_capture_mb,
            horizons=config.intervention.horizon_chunks,target_chunk=target,record_steps={(target,s) for s in range(count)},on_branch=on_branch)
        controller.finalize(probe)
        probe.finish(extra={"paired_branches":result.branch_summaries,"unrun_branch_ids":controller.unrun,
                            "reference_frozen_targets":len(controller.targets),"paired_rng":"per_call_and_final_matched",
                            "quality_metrics":"not_run; final latent differences are reference consistency, not quality"})
        final=result.last_output.payload
        return final if desired else final[0]
    except BaseException as exc:
        probe.finish(status="failed",extra={"error":f"{type(exc).__name__}: {exc}"});raise
    finally:probe.counterfactual=None
