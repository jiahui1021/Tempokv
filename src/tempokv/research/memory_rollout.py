"""E5 causal history-access comparisons with a labeled future-query oracle.

All branches regenerate their own clean history. Selection gathers attention
inputs and leaves the physical cache allocated: this is a memory-necessity
experiment, not an implementation of a smaller resident cache.
"""
from dataclasses import replace
import hashlib
import torch
from .archive import select_archive
from .attention_oracle import reference,_log_normalizer
from .metrics import relative_error
from .rollout import ReplayCase

POLICIES=('recent_only','uniform_archive','attention_archive','diversity_archive','oracle_archive')


def token_mass(q,k,key_chunk_size=1024):
    """FP32 scores only; the evaluated attention uses its native callback."""
    maximum,partition=_log_normalizer(q,k,key_chunk_size)
    result=[];query=q.float().transpose(1,2)
    for start in range(0,k.shape[1],key_chunk_size):
        logits=(query@k[:,start:start+key_chunk_size].float().permute(0,2,3,1))*q.shape[-1]**-.5
        probabilities=torch.exp((logits-maximum[...,None])-partition[...,None])
        result.append(probabilities.mean((0,1,2)))
    return torch.cat(result).cpu()


class MemoryController:
    def __init__(self,config):
        self.config=config;self.case=None;self.recorder=None;self.unrun=[];self.targets={};self.future={};self.future_bytes=0
        self.local_rows=[];self.limit=int(config.sampling.max_capture_mb*1024**2/4);self.query_capture_skips=0
    def setup_pass(self,case,recorder):self.case=case;self.recorder=recorder;self.local_rows=[]
    def wants(self,probe,layer):
        return probe.phase=='denoise' and layer in self.config.selection.layers and probe.chunk_id>=self.config.intervention.target_chunk
    def on_attention(self,probe,layer,q,k,v,metadata,attention_fn,baseline):
        if q.shape[0]!=1:raise ValueError('E5 paired histories require one sample per prompt/seed')
        probe._production_attention=attention_fn;output=baseline
        for head in self.config.selection.heads:
            window,_=probe.freeze_window(layer,head,0,q,k,v,metadata)
            ctx=window.context
            if self.case is None:
                ids=torch.linspace(0,q.shape[1]-1,min(q.shape[1],self.config.sampling.queries_per_head),device=q.device).round().long().unique()
                sample=q[:1,:,head:head+1].index_select(1,ids).detach().cpu().clone()
                if self.future_bytes+sample.numel()*sample.element_size()<=self.limit:
                    self.future.setdefault((layer,head),[]).append((ctx.chunk_id,ctx.denoise_step_index,sample))
                    self.future_bytes+=sample.numel()*sample.element_size()
                else:self.query_capture_skips+=1
                continue
            policy=self.case.metadata['policy'];retention=self.case.metadata['retention']
            if policy=='full_window':continue
            scores=oracle=None;oracle_scope=None
            if policy=='attention_archive':scores=token_mass(window.q_post,window.k_post,self.config.oracle.key_chunk_size)
            if policy=='oracle_archive':
                future=[value for chunk,step,value in self.future.get((layer,head),[]) if chunk>ctx.chunk_id and step==ctx.denoise_step_index]
                if not future:
                    oracle=token_mass(window.q_post,window.k_post,self.config.oracle.key_chunk_size)
                    oracle_scope='no_captured_future_query; current_Q_fallback_preserves_equal_budget'
                # A declared future-aware oracle sees reference Q only. Its keys
                # still come exclusively from this branch's already generated history.
                else:
                    oracle=sum(token_mass(value.to(k.device,k.dtype),window.k_post,self.config.oracle.key_chunk_size) for value in future)/len(future)
                    oracle_scope='bounded_reference_future_queries; diagnostic_not_optimal_oracle'
            chosen=select_archive(window,policy=policy,retention=retention,attention_scores=scores,oracle_scores=oracle,online=policy!='oracle_archive')
            index=torch.tensor(chosen['kept_indices'],device=k.device,dtype=torch.long)
            changed=replace(window,k_post=window.k_post.index_select(1,index),v=window.v.index_select(1,index),
                k_pre=None,pre_valid_mask=None,context=replace(ctx,key_abs_positions=ctx.key_abs_positions.index_select(0,index.to(ctx.key_abs_positions.device))))
            if len(index)==k.shape[1]:actual=baseline[:,:,head:head+1]
            else:actual=reference(changed,self.config.oracle.query_batch_size)
            if output is baseline:output=baseline.clone()
            output[:,:,head:head+1]=actual
            self.local_rows.append({**ctx.row(),**chosen,'strategy':policy,'retention':retention,
                'local_attention_relative_error':relative_error(baseline[:,:,head:head+1],actual),'status':'measured',
                'metric_scope':'causal_history_attention_access','online_policy':policy!='oracle_archive','oracle_scope':oracle_scope})
        return output
    def cases(self,trace):
        cases=[ReplayCase('E5_full_window',(),kind='archive',metadata={'policy':'full_window','retention':1.})]
        for retention in self.config.compression.retention:
            for policy in POLICIES:
                identity=hashlib.sha256(repr((policy,retention)).encode()).hexdigest()[:16]
                cases.append(ReplayCase(identity,(),kind='archive',metadata={'policy':policy,'retention':retention}))
        self.unrun=[case.case_id for case in cases[self.config.intervention.max_branches:]]
        return cases[:self.config.intervention.max_branches]
    def record_branch(self,probe,case,trace,output,rows):
        base={'prompt_index':probe.prompt_index,'seed':probe.seed,'sample_id':0,'paired_run_id':case.case_id,
              'strategy':case.metadata['policy'],'retention':case.metadata['retention'],'metric_scope':'paired_generation',
              'causal_scope':'branch_own_generated_history; current_Q_scoring_allowed; no_future_except_labeled_oracle',
              'phase':'denoise','intervention':'archive','evaluation_status':'exploratory','quality_status':'unknown',
              'feature_consistency_status':'not_measured_requires_declared_entity_state_features',
              'storage_contract':'attention-input gather only; full physical cache retained',
              'paired_rng':'matched_per_call_and_final_state'}
        for row in self.local_rows:
            probe.store.write('E5',{**base,**row,'policy_id':case.metadata['policy']})
        for row in rows:
            if 'horizon_chunks' in row:probe.store.write('E5',{**base,**row})
        if case.metadata['policy']=='full_window':
            probe.store.write('E5',{**base,'local_attention_relative_error':0.,'actual_retention':1.,'clean_latent_relative_error':0.})
    def finalize(self,probe):
        probe.store.write('E5',{'strategy':'memory_protocol','statistic':'capture_accounting','future_query_bytes':self.future_bytes,
            'metric_scope':'capture_accounting',
            'future_query_budget_bytes':self.limit,'skipped_future_query_captures':self.query_capture_skips,
            'status':'not_measured','reason':'entity/state visual features and real reappearance videos not evaluated by latent replay',
            'oracle_scope':'bounded_sampled_reference_future_queries_only; unavailable events explicitly skipped'})
