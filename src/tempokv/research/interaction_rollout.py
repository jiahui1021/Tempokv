"""Single-event paired E7 2x2; decoded histories are never mixed across runs."""
from dataclasses import replace
import torch
from .archive import select_archive
from .attention_oracle import evaluate
from .counterfactual import CounterfactualController,case_id
from .diagnostics import quantized_variant
from .interventions import identity,KVVariant
from .memory_rollout import token_mass
from .metrics import relative_error
from .rollout import FrozenEvent,ReplayCase


class InteractionController(CounterfactualController):
    def __init__(self,config):
        super().__init__(config);self.masks={};self.measured={}
    def on_attention(self,probe,layer,q,k,v,metadata,attention_fn,baseline):
        c=self.config.intervention
        if probe.step_index!=c.target_step:return baseline
        probe._production_attention=attention_fn
        window,_=probe.freeze_window(layer,c.target_head,0,q,k,v,metadata);ctx=window.context
        key=(layer,c.target_head,probe.chunk_id,probe.step_index)
        if self.case is None:
            eligible=tuple(ctx.key_abs_positions[ctx.eligible_mask].tolist())
            self.targets[0]=FrozenEvent(*key,eligible)
            self.reference_contexts[key]=ctx.row()
            scores=token_mass(window.q_post,window.k_post,self.config.oracle.key_chunk_size)
            for retention in self.config.compression.retention:
                selection=select_archive(window,policy='attention_archive',retention=retention,attention_scores=scores)
                self.masks[retention]=tuple(ctx.key_abs_positions[list(selection['kept_indices'])].tolist())
            return baseline
        label=self.case.metadata['cell'];retention=self.case.metadata['retention'];bits=self.case.metadata['bits']
        if label=='A':return baseline
        lookup={int(p):i for i,p in enumerate(ctx.key_abs_positions.tolist())}
        event=self.case.events[0]
        if not all(p in lookup for p in event.absolute_tokens):raise ValueError('E7 frozen eligible positions missing in branch')
        indices=[lookup[p] for p in event.absolute_tokens]
        if any(ctx.protected_mask[i] for i in indices):raise ValueError('E7 protected position overlap')
        variant=identity(window);costs={}
        if label in {'C','D'}:variant,costs,_,_=quantized_variant(window,indices,*bits,self.config.compression.group_size)
        if label in {'B','D'}:
            kept=self.masks[retention]
            if not all(p in lookup for p in kept):raise ValueError('E7 frozen pruning mask absent')
            index=torch.tensor([lookup[p] for p in kept],device=k.device)
            variant=KVVariant(variant.k_post.index_select(1,index),variant.v.index_select(1,index),
                variant.coordinates.index_select(0,index.to(variant.coordinates.device)),variant.log_count.index_select(0,index.to(variant.log_count.device)),
                tuple(variant.provenance[i] for i in index.tolist()),variant.details)
        measurement=evaluate(window,k=variant.k_post,v=variant.v,query_batch_size=self.config.oracle.query_batch_size)
        actual=measurement['output'];result=baseline.clone();result[:,:,c.target_head:c.target_head+1]=actual
        self.recorder.record_injection(event)
        self.local_rows=[{**ctx.row(),**costs,'local_attention_relative_error':relative_error(baseline[:,:,c.target_head:c.target_head+1],actual),
            'actual_retained_tokens':variant.token_count,'actual_retention':variant.token_count/k.shape[1],
            'mask_protocol':'frozen_BF16_reference_mask_for_B_and_D','frozen_absolute_positions':list(self.masks[retention])}]
        return result
    def cases(self,trace):
        if not self.targets:return [ReplayCase('E7_A_only',(),metadata={'cell':'A','retention':1.,'bits':(16,16)})]
        event=self.targets[0];cases=[]
        pairs=tuple(dict.fromkeys(((4,4),*self.config.compression.mixed_bits)))
        for retention in self.config.compression.retention:
            for bits in pairs:
                for cell in ('A','B','C','D'):
                    cases.append(ReplayCase(case_id('E7',retention,bits,cell),() if cell=='A' else (event,),kind='joint',
                        metadata={'cell':cell,'retention':retention,'bits':bits}))
        self.unrun=[case.case_id for case in cases[self.config.intervention.max_branches:]]
        return cases[:self.config.intervention.max_branches]
    def record_branch(self,probe,case,trace,output,rows):
        c=self.config.intervention;cell=case.metadata['cell'];retention=case.metadata['retention'];kb,vb=case.metadata['bits']
        context=self.reference_contexts.get((c.target_layer,c.target_head,c.target_chunk,c.target_step),{})
        local=self.local_rows[0] if self.local_rows else {'local_attention_relative_error':0.}
        for row in rows:
            if 'horizon_chunks' not in row:continue
            record={**context,**local,**row,'strategy':{'A':'A_exact','B':'B_pruning_only','C':'C_quant_only','D':'D_joint'}[cell],
                'retention':retention,'k_bits':kb,'v_bits':vb,'paired_run_id':case.case_id,'metric_scope':'paired_generation',
                'paired_rng':'matched_per_call_and_final_state','compression_scope':'single_frozen_attention_event',
                'quantizer':'independent_symmetric_fake_quant_v1','jensen_correction':False}
            probe.store.write('E7',record)
            if row.get('horizon_chunks')==0 and row.get('status')=='measured':self.measured[retention,kb,vb,cell]=record
    def finalize(self,probe):
        for retention,kb,vb in sorted({key[:3] for key in self.measured}):
            cells={cell:self.measured.get((retention,kb,vb,cell)) for cell in 'ABCD'}
            if not all(cells.values()):continue
            a,b,c,d=[cells[cell]['clean_latent_relative_error'] for cell in 'ABCD']
            probe.store.write('E7',{**cells['D'],'strategy':'paired_clean_interaction','interaction':d-c-b+a,'D_minus_C':d-c,'B_minus_A':b-a,
                'interpretation':'nonlinear_error_contrast_not_physical_synergy'})


def make_controller(config):
    if 'E5' in config.experiments:
        if set(config.experiments)!={'E5'}:raise ValueError('E5 causal archive replay must run separately from single-event experiments')
        from .memory_rollout import MemoryController
        return MemoryController(config)
    if 'E7' in config.experiments:
        if set(config.experiments)!={'E7'}:raise ValueError('E7 paired 2x2 must run separately from other interventions')
        return InteractionController(config)
    return CounterfactualController(config)
