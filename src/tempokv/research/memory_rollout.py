"""Persistent E5 archives, enforced for denoising and clean cache refresh.

Only retained positions and newly generated positions can enter a later choice.
The native backing remains allocated for diagnostic comparisons, but evicted
rows cannot influence scores or branch attention. This is a logical archive
experiment and makes no resident-memory saving claim.
"""
from dataclasses import replace
import hashlib
import torch
from .archive import select_archive
from .attention_oracle import reference,_log_normalizer
from .counterfactual import exact_keep
from .metrics import relative_error
from .rollout import ReplayCase

POLICIES=('recent_only','uniform_archive','attention_archive','diversity_archive','oracle_archive')
PROTOCOL='research.e5.persistent.v2'


def token_mass(q,k,key_chunk_size=1024):
    """FP32 scoring only; evaluated attention uses its native callback."""
    maximum,partition=_log_normalizer(q,k,key_chunk_size)
    result=[];query=q.float().transpose(1,2)
    for start in range(0,k.shape[1],key_chunk_size):
        logits=(query@k[:,start:start+key_chunk_size].float().permute(0,2,3,1))*q.shape[-1]**-.5
        result.append(torch.exp((logits-maximum[...,None])-partition[...,None]).mean((0,1,2)))
    return torch.cat(result).cpu()


def _gather(window,indices):
    index=torch.as_tensor(indices,device=window.k_post.device,dtype=torch.long)
    return replace(window,k_post=window.k_post.index_select(1,index),v=window.v.index_select(1,index),
        k_pre=None,pre_valid_mask=None,context=replace(window.context,
        key_abs_positions=window.context.key_abs_positions.index_select(0,index.to(window.context.key_abs_positions.device))))


class MemoryController:
    def __init__(self,config):
        self.config=config;self.case=None;self.recorder=None;self.unrun=[];self.targets={};self.future={};self.future_bytes=0
        self.local_rows=[];self.limit=int(config.sampling.max_capture_mb*1024**2/4);self.query_capture_skips=0
        self.archives={}
    def setup_pass(self,case,recorder):
        self.case=case;self.recorder=recorder;self.local_rows=[];self.archives={}
    def wants(self,probe,layer):
        selected=self.config.intervention.e5_scope=='all' or layer in self.config.selection.layers
        return probe.phase in {'denoise','clean_refresh'} and selected and probe.chunk_id>=self.config.intervention.target_chunk
    def _sample_q(self,q):
        ids=torch.linspace(0,q.shape[1]-1,min(q.shape[1],self.config.sampling.queries_per_head),device=q.device).round().long().unique()
        return q.index_select(1,ids)
    def on_attention(self,probe,layer,q,k,v,metadata,attention_fn,baseline):
        if q.shape[0]!=1:raise ValueError('E5 paired histories require one sample per prompt/seed')
        probe._production_attention=attention_fn;output=baseline
        heads=range(q.shape[2]) if self.config.intervention.e5_scope=='all' else self.config.selection.heads
        for head in heads:
            window,_=probe.freeze_window(layer,head,0,q,k,v,metadata);ctx=window.context
            if self.case is None:
                sample=self._sample_q(window.q_post).detach().cpu().clone()
                if self.future_bytes+sample.numel()*sample.element_size()<=self.limit:
                    self.future.setdefault((layer,head),[]).append((ctx.chunk_id,ctx.phase,ctx.denoise_step_index,sample))
                    self.future_bytes+=sample.numel()*sample.element_size()
                else:self.query_capture_skips+=1
                continue
            policy=self.case.metadata['policy'];retention=self.case.metadata['retention']
            if policy=='full_window':
                self.local_rows.append({**ctx.row(),'strategy':policy,'retention':1.,
                    'actual_retained_tokens':len(ctx.key_abs_positions),'actual_retention':1.,
                    'local_attention_relative_error':0.,'total_token_accesses':len(ctx.key_abs_positions)*q.shape[1],
                    'metric_scope':'full_history_reference_access','status':'measured','archive_protocol':PROTOCOL,
                    'cost_scope':'full_history_key_accesses_including_clean_refresh'})
                continue
            previous=self.archives.get((layer,head));positions=ctx.key_abs_positions.detach().cpu()
            if previous is None:
                allowed=torch.ones(len(positions),dtype=torch.bool);new_chunk=True
            else:
                if ctx.current_chunk_start<previous['chunk_start']:raise ValueError('E5 archive events must be chronological within a replay')
                allowed=torch.isin(positions,previous['positions']) | (positions>=previous['seen_end'])
                new_chunk=ctx.current_chunk_start!=previous['chunk_start']
            if not bool(allowed[ctx.protected_mask.cpu()].all()):
                raise ValueError('E5 cannot restore previously evicted positions through a changed protection rule')
            available=_gather(window,allowed.nonzero().flatten())
            requested=exact_keep(int(ctx.eligible_mask.sum()),retention)
            available_count=int(available.context.eligible_mask.sum())
            budget=min(requested,available_count)
            oracle_scope=None;scoring_queries=0
            if new_chunk:
                scores=oracle=None;sample_q=self._sample_q(available.q_post)
                if policy=='attention_archive':
                    scores=token_mass(sample_q,available.k_post,self.config.oracle.key_chunk_size);scoring_queries=sample_q.shape[1]
                if policy=='oracle_archive':
                    future=[value for chunk,phase,step,value in self.future.get((layer,head),[])
                            if chunk>ctx.chunk_id and phase==ctx.phase and step==ctx.denoise_step_index]
                    if future:
                        oracle=sum(token_mass(value.to(k.device,k.dtype),available.k_post,self.config.oracle.key_chunk_size) for value in future)/len(future)
                        scoring_queries=sum(value.shape[1] for value in future)
                        oracle_scope='bounded_reference_future_queries_on_available_archive; diagnostic_not_optimal'
                    else:
                        oracle=token_mass(sample_q,available.k_post,self.config.oracle.key_chunk_size);scoring_queries=sample_q.shape[1]
                        oracle_scope='no_captured_future_query; current_Q_fallback_preserves_budget'
                chosen=select_archive(available,policy=policy,retention=retention,attention_scores=scores,oracle_scores=oracle,
                                      online=policy!='oracle_archive',token_budget=budget)
            else:
                # The mask is frozen across all denoise steps and clean refresh
                # in this chunk. Reading a dropped row to reselect it is forbidden.
                chosen=select_archive(available,policy=policy,retention=1.,online=policy!='oracle_archive',token_budget=available_count)
                oracle_scope='same_chunk_frozen_archive_mask' if policy=='oracle_archive' else None
            changed=_gather(available,chosen['kept_indices'])
            retained=changed.context.key_abs_positions.detach().cpu().clone()
            self.archives[layer,head]={'positions':retained,'chunk_start':ctx.current_chunk_start,
                'seen_end':ctx.current_chunk_start+ctx.grid_t*ctx.frame_tokens}
            if len(retained)==k.shape[1]:actual=baseline[:,:,head:head+1]
            else:actual=reference(changed,self.config.oracle.query_batch_size)
            if output is baseline:output=baseline.clone()
            output[:,:,head:head+1]=actual
            index_by_absolute={int(p):i for i,p in enumerate(positions.tolist())}
            kept=tuple(index_by_absolute[int(p)] for p in retained.tolist());kept_set=set(kept)
            access_count=len(retained)*window.q_post.shape[1]
            self.local_rows.append({**ctx.row(),**chosen,'strategy':policy,'retention':retention,
                'kept_indices':kept,'removed_indices':tuple(i for i in range(len(positions)) if i not in kept_set),
                'kept_abs_positions':retained.tolist(),'key_token_count':len(positions),'actual_retained_tokens':len(retained),
                'actual_retention':len(retained)/len(positions),'original_eligible_tokens':int(ctx.eligible_mask.sum()),
                'available_eligible_tokens':available_count,'requested_eligible_budget':requested,'budget_underfill':requested-chosen['eligible_kept_tokens'],
                'local_attention_relative_error':relative_error(baseline[:,:,head:head+1],actual),'status':'measured',
                'metric_scope':'causal_history_attention_access','online_policy':policy!='oracle_archive','oracle_scope':oracle_scope,
                'archive_protocol':PROTOCOL,'intervention_scope':self.config.intervention.e5_scope,'archive_mask_reused':not new_chunk,
                'scoring_query_samples':scoring_queries,'total_token_accesses':access_count,
                'cost_scope':'actual_selected_attention_key_accesses_including_clean_refresh; scoring_separate',
                'budget_cohort':f'tokens:{len(retained)}','causal_scope':'retained_history_and_new_current_only; no_evicted_token_reselection',
                'physical_kv_bytes':window.k_post.untyped_storage().nbytes()+window.v.untyped_storage().nbytes(),
                'compute_only_full_backing_bytes':window.k_post.untyped_storage().nbytes()+window.v.untyped_storage().nbytes(),
                'reference_scope':'diagnostic_full_backing_at_current_branch_state; never_used_for_policy_selection'})
        return output
    def cases(self,trace):
        cases=[ReplayCase('E5_full_window',(),kind='archive',metadata={'policy':'full_window','retention':1.})]
        for retention in self.config.compression.retention:
            for policy in POLICIES:
                identity=hashlib.sha256(repr((PROTOCOL,policy,retention,self.config.intervention.e5_scope)).encode()).hexdigest()[:16]
                cases.append(ReplayCase(identity,(),kind='archive',metadata={'policy':policy,'retention':retention}))
        self.unrun=[case.case_id for case in cases[self.config.intervention.max_branches:]]
        return cases[:self.config.intervention.max_branches]
    def record_branch(self,probe,case,trace,output,rows):
        base={'prompt_index':probe.prompt_index,'seed':probe.seed,'sample_id':0,'paired_run_id':case.case_id,
              'strategy':case.metadata['policy'],'retention':case.metadata['retention'],'metric_scope':'paired_generation',
              'archive_protocol':PROTOCOL,'intervention_scope':self.config.intervention.e5_scope,
              'causal_scope':'branch_own_generated_history; persistent_archive_on_denoise_and_clean',
              'phase':'denoise','intervention':'archive','evaluation_status':'exploratory','quality_status':'unknown',
              'feature_consistency_status':'not_measured_requires_declared_entity_state_features',
              'storage_contract':'logical_persistent_archive; full_physical_backing_retained_for_diagnostic_reference',
              'paired_rng':'matched_per_call_and_final_state'}
        for row in self.local_rows:probe.store.write('E5',{**base,**row,'policy_id':case.metadata['policy']})
        for row in rows:
            if 'horizon_chunks' in row:probe.store.write('E5',{**base,**row})
        probe.store.write('E5',{**base,'statistic':'complete_archive_access_cost',
            'total_token_accesses':sum(row['total_token_accesses'] for row in self.local_rows),
            'denoise_token_accesses':sum(row['total_token_accesses'] for row in self.local_rows if row['phase']=='denoise'),
            'clean_token_accesses':sum(row['total_token_accesses'] for row in self.local_rows if row['phase']=='clean_refresh'),
            'status':'measured' if self.local_rows else 'not_measured'})
        if case.metadata['policy']=='full_window':
            probe.store.write('E5',{**base,'local_attention_relative_error':0.,'actual_retention':1.,'clean_latent_relative_error':0.})
    def finalize(self,probe):
        probe.store.write('E5',{'strategy':'memory_protocol','statistic':'capture_accounting','future_query_bytes':self.future_bytes,
            'metric_scope':'capture_accounting','archive_protocol':PROTOCOL,
            'archive_position_state_bytes':sum(state['positions'].numel()*state['positions'].element_size() for state in self.archives.values()),
            'future_query_budget_bytes':self.limit,'skipped_future_query_captures':self.query_capture_skips,
            'status':'not_measured','reason':'real_entity/state_features_require_explicit_evaluation; no_quality_inferred',
            'oracle_scope':'bounded_reference_future_queries_only; absent_future_uses_labeled_causal_fallback'})
