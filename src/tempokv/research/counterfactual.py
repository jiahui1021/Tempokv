"""Frozen event interventions for paired full-inference replay (M2)."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from decimal import Decimal,ROUND_HALF_UP
import torch
from .capture import stable_seed
from .context import FrozenAttention
from . import interventions as ops
from .attention_oracle import evaluate,leave_one_out
from .metrics import relative_error
from .rollout import FrozenEvent,ReplayCase
from .calibration import calibrate_local_error,relative_rmse_intervention,calibration_metadata


def event_key(event):return (event.layer,event.head,event.chunk,event.step,event.absolute_tokens)

def case_id(*parts):return hashlib.sha256(repr(parts).encode()).hexdigest()[:20]

def exact_keep(count,retention):return int((Decimal(count)*Decimal(str(retention))).to_integral_value(rounding=ROUND_HALF_UP))


class CounterfactualController:
    def __init__(self,config):
        self.config=config;self.case=None;self.recorder=None;self.targets={};self.details={};self.calibrated={};self.step_events={};self.step_costs={}
        self.local_rows=[];self.reference_contexts={};self.unrun=[];self.e6_measurements=[]

    def setup_pass(self,case,recorder):self.case=case;self.recorder=recorder;self.local_rows=[]

    def wants(self,probe,layer):
        c=self.config.intervention
        return probe.phase=="denoise" and layer==c.target_layer and probe.chunk_id==c.target_chunk

    def on_attention(self,probe,layer,q,k,v,metadata,attention_fn,baseline):
        if not self.wants(probe,layer):return baseline
        c=self.config.intervention;head=c.target_head
        if q.shape[0]!=1:raise ValueError("paired counterfactual replay currently requires one sample per prompt/seed")
        probe._production_attention=attention_fn
        window,candidates=probe.freeze_window(layer,head,0,q,k,v,metadata)
        ctx=window.context;step=probe.step_index;key=(layer,head,probe.chunk_id,step)
        if self.case is None:
            self.reference_contexts[key]=ctx.row()
            # E6 freezes the complete eligible historical set, not just the sampled raw reservoir.
            eligible=ctx.key_abs_positions[ctx.eligible_mask].tolist()
            self.step_events[step]=FrozenEvent(layer,head,probe.chunk_id,step,tuple(eligible))
            self.step_costs[step]=(k.shape[1],int(ctx.protected_mask.sum()),q.shape[1])
            if step!=c.target_step or not candidates:return baseline
            chosen=candidates[c.target_block_rank:c.target_block_rank+c.max_branches]
            if not chosen:return baseline
            measurement=leave_one_out(window,[indices for _,indices in chosen],query_batch_size=self.config.oracle.query_batch_size,
                                      block_batch_size=self.config.oracle.block_batch_size,key_chunk_size=self.config.oracle.key_chunk_size)
            for rank,((entry,indices),local) in enumerate(zip(chosen,measurement)):
                absolute=tuple(ctx.key_abs_positions[list(indices)].tolist())
                event=FrozenEvent(layer,head,probe.chunk_id,step,absolute)
                self.targets[rank]=event
                self.details[event_key(event)]={**ctx.row(),"block_id":f"frame{entry.frame}_spatialblock{entry.block_id}",
                    "attention_mass":local["attention_mass"],"reference_local_sample_scope":"full_current_query_chunk", "frame_age":ctx.current_chunk_start//ctx.frame_tokens-entry.frame}
                for strength in (0.,*c.strengths):
                    if c.kind in {"drop","sparse","none"}:continue
                    seed=stable_seed(ctx.identity,"counterfactual_noise",strength,c.kind,c.role)
                    if c.calibration=="local_d_att":
                        calibrated=calibrate_local_error(window,indices,kind=c.kind,role=c.role,target_error=strength,seed=seed,
                                                         query_batch_size=self.config.oracle.query_batch_size)
                    else:calibrated=relative_rmse_intervention(window,indices,strength,kind=c.kind,role=c.role,seed=seed)
                    # Only scalar protocol metadata persists past this event.
                    data=calibration_metadata(calibrated)
                    data.update(noise_seed=seed,target_nmse=calibrated.get("target_nmse",strength**2))
                    self.calibrated[event_key(event),strength]=data
            return baseline
        matching=[event for event in self.case.events if (event.layer,event.head,event.chunk,event.step)==key]
        if not matching:return baseline
        if len(matching)!=1:raise ValueError("at most one intervention per layer/head/step is allowed")
        event=matching[0]
        lookup={int(pos):i for i,pos in enumerate(ctx.key_abs_positions.tolist())}
        if not all(pos in lookup for pos in event.absolute_tokens):raise ValueError("frozen reference positions absent in branch window")
        indices=tuple(lookup[pos] for pos in event.absolute_tokens)
        if any(ctx.protected_mask[i] for i in indices):raise ValueError("frozen intervention overlaps current/recent/sink protection")
        if self.case.kind=="sparse":
            retention=self.case.metadata["retentions"][step]
            keep=exact_keep(len(indices),retention)
            if keep:
                import numpy as np
                chosen=set(np.linspace(0,len(indices)-1,keep).round().astype(int).tolist())
                remove=tuple(index for offset,index in enumerate(indices) if offset not in chosen)
            else:remove=indices
            variant=ops.drop(window,remove);is_null=not remove
        elif self.case.kind=="drop":variant=ops.drop(window,indices);is_null=not indices
        elif self.case.kind=="none":variant=ops.identity(window);is_null=True
        else:
            params=self.case.metadata["calibration"]
            variant=ops.perturb(window,indices,kind=self.case.kind,role=self.case.role,target_nmse=params["target_nmse"],seed=params["noise_seed"])
            is_null=params["target_nmse"]==0
        if is_null:changed=baseline[:,:,head:head+1];discrepancy=0.
        else:
            measured=evaluate(window,k=variant.k_post,v=variant.v,log_count=variant.log_count,query_batch_size=self.config.oracle.query_batch_size)
            changed=measured["output"]
            discrepancy=relative_error(baseline[:,:,head:head+1],measured["reference_output"])
        local_error=relative_error(baseline[:,:,head:head+1],changed)
        output=baseline.clone() if not is_null else baseline
        if not is_null:output[:,:,head:head+1]=changed
        self.recorder.record_injection(event)
        self.local_rows.append({**ctx.row(),"case_id":self.case.case_id,"local_attention_relative_error":local_error,
            "query_scope":"full_current_chunk","single_head_kernel_reference_discrepancy":discrepancy,
            "actual_retained_tokens":variant.token_count,"actual_retention":variant.token_count/k.shape[1],
            "frozen_absolute_positions":list(event.absolute_tokens),"intervention_hits":1,"calibration_protocol":c.calibration})
        return output

    def cases(self,reference_trace):
        c=self.config.intervention
        cases=[ReplayCase(case_id("null",self.config.signature),(),kind="none",metadata={"control":"zero_intervention"})]
        events=list(self.targets.values())
        if events:
            if c.kind == "drop":
                for event in events:cases.append(ReplayCase(case_id(event_key(event),c.kind),(event,),kind=c.kind,role=c.role,strength=1.))
            elif c.kind != "none":
                event=events[0]
                for strength in c.strengths:
                    calibration=self.calibrated.get((event_key(event),strength))
                    if calibration is not None:cases.append(ReplayCase(case_id(event_key(event),c.kind,strength),(event,),kind=c.kind,role=c.role,
                        strength=strength,metadata={"calibration":calibration}))
        if "E6" in self.config.experiments:
            steps=sorted(self.step_events)
            for step in steps:
                if not self.step_events[step].absolute_tokens:continue
                retention=[1.]*len(steps);retention[step]=.5
                cases.append(ReplayCase(case_id("single_step",step),(self.step_events[step],),kind="sparse",strength=.5,
                                       metadata={"retentions":retention,"policy_kind":"single_step","step":step}))
            for policy in c.e6_step_policies:
                if len(policy)!=len(steps):raise ValueError("E6 retention policy length must match the actual denoise schedule")
                selected=tuple(self.step_events[step] for step in steps)
                cases.append(ReplayCase(case_id("step_policy",policy),selected,kind="sparse",metadata={"retentions":list(policy),"policy_kind":"complete_step_policy"}))
        if len(cases)>c.max_branches:
            self.unrun=[case.case_id for case in cases[c.max_branches:]]
            cases=cases[:c.max_branches]
        return cases

    def record_branch(self,probe,case,trace,output,rows):
        c=self.config.intervention
        context=self.reference_contexts.get((c.target_layer,c.target_head,c.target_chunk,c.target_step),{})
        first=case.events[0] if case.events else None
        event_details=self.details.get(event_key(first),{}) if first else {}
        prefix={**context,**event_details,"prompt_index":probe.prompt_index,"seed":probe.seed,"sample_id":0,"paired_run_id":case.case_id,
            "strategy":case.kind,"intervention":case.kind,"strength":case.strength,"requested_strength":case.strength,
            "role":case.role,
            "calibration_protocol":c.calibration,"case_id":case.case_id,"metric_scope":"paired_generation",
            "layer":first.layer if first else c.target_layer,"head":first.head if first else c.target_head,
            "chunk_index":first.chunk if first else c.target_chunk,"denoise_step_index":first.step if first else c.target_step,
            "phase":"denoise","query_scope":"full_current_chunk","paired_rng":"matched_per_call_and_final_state"}
        calibration=case.metadata.get("calibration",{})
        prefix.update(calibration_matched=calibration.get("matched",True if not case.events or case.kind=="drop" else None),
                      calibration_details=calibration,
                      comparison_cohort="matched" if calibration.get("matched",True) else "unmatched_calibration; exclude_from_equal_error_comparison")
        local=self.local_rows[0]["local_attention_relative_error"] if self.local_rows else 0. if not case.events else None
        step_error=next((r.get("model_output_relative_error") for r in rows if r.get("model_output_relative_error") is not None
                         and (first is None or r.get("denoise_step_index")==first.step)),None)
        clean_error=next((r.get("clean_latent_relative_error") for r in rows if r.get("horizon_chunks")==0 and r.get("clean_latent_relative_error") is not None),None)
        if "E1" in self.config.experiments:
            probe.store.write("E1",{**prefix,"local_attention_relative_error":local,"model_output_relative_error":step_error,
                "clean_latent_relative_error":clean_error,"horizon_chunks":0,
                "attention_mass":self.details.get(event_key(first),{}).get("attention_mass") if first else None,
                "status":"measured" if clean_error is not None else "not_measured"})
        if "E2" in self.config.experiments:
            for row in rows:
                if "horizon_chunks" in row:probe.store.write("E2",{**prefix,**row,"local_attention_relative_error":local})
        if "E6" in self.config.experiments and case.kind in {"sparse","none"}:
            retentions=case.metadata.get("retentions",[1.]*len(self.step_costs))
            actual=[protected+exact_keep(tokens-protected,r) for (tokens,protected,_),r in zip([self.step_costs[s] for s in sorted(self.step_costs)],retentions)]
            record={**prefix,"strategy":case.metadata.get("policy_kind","null"),"local_attention_relative_error":local,
                "clean_latent_relative_error":clean_error,"retention":sum(actual)/sum(x[0] for x in self.step_costs.values()) if self.step_costs else None,
                "total_token_accesses":sum(actual),"actual_tokens_per_step":actual,"retentions":retentions,
                "cost_scope":"target_layer_head_key_accesses; clean_refresh excluded","budget_cohort":f"accesses:{sum(actual)}",
                "local_errors_by_step":self.local_rows,
                "status":"measured" if clean_error is not None else "not_measured"}
            probe.store.write("E6",record)
            if clean_error is not None:self.e6_measurements.append(record)

    def finalize(self,probe):
        measured=getattr(self,"e6_measurements",[])
        if not measured:return
        for budget in sorted({row["total_token_accesses"] for row in measured}):
            eligible=[row for row in measured if row["total_token_accesses"]<=budget and row.get("clean_latent_relative_error") is not None]
            if not eligible:continue
            best=min(eligible,key=lambda row:(row["clean_latent_relative_error"],row["total_token_accesses"],row["paired_run_id"]))
            probe.store.write("E6",{**best,"strategy":"measured_step_budget_oracle","budget_cohort":f"budget_accesses:{budget}",
                "requested_access_budget":budget,"oracle_scope":"supplied_replayed_schedules_only",
                "objective_kind":"measured_complete_schedule_clean_latent_error","equality_contract":"actual accesses, protected tokens fixed"})
