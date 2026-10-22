"""M1 frozen-window diagnostics; every treatment returns to the same reference."""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from . import attention_oracle as oracle
from . import interventions as ops
from .capture import stable_seed
from .metrics import comparable_correlations, nmse
from .precision import fake_quantize, PrecisionOption, allocation_frontier, precision_policy_rows, allocate
from .scores import candidate_scores, cosine


def evaluate_variant(window,variant,config):
    return oracle.evaluate(window,k=variant.k_post,v=variant.v,log_count=variant.log_count,
                           query_batch_size=config.oracle.query_batch_size)


def write_local(window,candidates,store,config):
    ctx=window.context;base=ctx.row(); blocks=[indices for _,indices in candidates]
    for exp in ("S2","E1","E3","E4"):
        if exp in config.experiments:
            store.write(exp,{**base,"block_id":"null","strategy":"null","intervention":"none","metric_scope":"attention_local",
                             "local_attention_relative_error":0.0,"actual_retained_tokens":window.k_post.shape[1],"actual_retention":1.0,
                             "budget_cohort":f"tokens:{window.k_post.shape[1]}"})
    if not blocks:
        for exp in set(config.experiments)&{"S2","E1","E3","E4"}:
            store.write(exp,{**base,"status":"not_measured","reason":"no_eligible_clean_history_blocks"})
        return {"candidates":[],"oracle":[],"scores":{},"nearest":{}}
    rows=oracle.leave_one_out(window,blocks,query_batch_size=config.oracle.query_batch_size,
        block_batch_size=config.oracle.block_batch_size,key_chunk_size=config.oracle.key_chunk_size)
    scores,nearest=candidate_scores(window,candidates,rows)
    for i,((entry,indices),row) in enumerate(zip(candidates,rows)):
        ident=f"frame{entry.frame}_spatialblock{entry.block_id}"
        raw={**base,"block_id":ident,"history_frame":entry.frame,"frame_age":ctx.current_chunk_start//ctx.frame_tokens-entry.frame,
             "key_abs_positions":ctx.key_abs_positions[list(indices)].tolist(),"valid_key_mask":"all_true_no_padding",
             "local_attention_relative_error":row["local_attention_relative_error"],"attention_mass":row["attention_mass"],
             "candidate_scope":"sampled_clean_block_reservoir","candidate_count":len(blocks),"metric_scope":"attention_local",
             "actual_retained_tokens":window.k_post.shape[1]-len(indices),"actual_retention":1-len(indices)/window.k_post.shape[1],
             "intervention":"drop","query_scope":"sampled_queries_full_actual_KV_window"}
        if "S2" in config.experiments:
            for name,values in scores.items():store.write("S2",{**raw,"score_type":name,"score":float(values[i])})
        if "E1" in config.experiments:
            store.write("E1",{**raw,"score_type":"attention_mass","score":row["attention_mass"],
                              "model_output_relative_error":None,"clean_latent_relative_error":None})
    if "S2" in config.experiments:
        correlations=comparable_correlations(scores,[r["local_attention_relative_error"] for r in rows])
        for row in correlations:store.write("S2",{**base,**row,"statistic":"candidate_rank_correlation","metric_scope":"attention_local"})
    if "E3" in config.experiments:_substitutability(window,candidates,rows,nearest,store,config)
    if "E4" in config.experiments:_asymmetry(window,blocks,store,config)
    return {"candidates":candidates,"oracle":rows,"scores":scores,"nearest":nearest}


def _substitutability(window,candidates,oracle_rows,nearest,store,config):
    base=window.context.row(); eligible_pairs=[]
    for i,(a,ia) in enumerate(candidates):
        choices=[]
        for j,(b,ib) in enumerate(candidates):
            if i==j or len(ia)!=len(ib) or abs(a.frame-b.frame)>config.sampling.max_pair_frame_distance:continue
            ck,cv=cosine(a.k_pre,b.k_pre),cosine(a.v,b.v)
            if np.isfinite(ck+cv):choices.append((-(ck+cv)/2,i,j,ck,cv))
        if choices:eligible_pairs.append(min(choices))
    chosen=sorted(eligible_pairs)[:config.sampling.max_pairs]
    for _,i,j,ck,cv in chosen:
        a,ia=candidates[i];b,ib=candidates[j]
        pair_id=f"{a.frame}:{a.block_id}|{b.frame}:{b.block_id}"
        groups=list(zip(ia,ib))
        cases=(("drop",ops.drop(window,ia)),("replace",ops.replace(window,ia,ib)),
               ("average_merge",ops.merge(window,groups,multiplicity=False)),
               ("weighted_merge",ops.merge(window,groups,multiplicity=True)))
        for name,variant in cases:
            measurement=evaluate_variant(window,variant,config)
            store.write("E3",{**base,"block_id":pair_id,"strategy":name,"intervention":name,
                "local_attention_relative_error":measurement["relative_error"],"actual_retained_tokens":variant.token_count,
                "actual_retention":variant.token_count/window.k_post.shape[1],"budget_cohort":f"tokens:{variant.token_count}",
                "k_cosine":ck,"v_cosine":cv,"joint_similarity":(ck+cv)/2,"temporal_distance":abs(a.frame-b.frame),
                "representative_positions":variant.coordinates.tolist(),"backend_protocol":measurement["backend_protocol"],
                "metric_scope":"attention_local","approximation":"pre_rope_mean_then_rotate; not_lossless_in_general",
                "grouping_protocol":"exhaustive_equal_size_nearest_joint_cosine_with_distance_limit"})
        # Conditions are fixed original candidate sets; target effects are measured
        # relative to each conditioned reference, never a sum of LOO effects.
        other=next((indices for n,(_,indices) in enumerate(candidates) if n not in {i,j}),())
        for condition in ((),other):
            measurement=oracle.conditional_drop(window,ia,condition,query_batch_size=config.oracle.query_batch_size)
            store.write("E3",{**base,"block_id":pair_id,"strategy":"conditional_drop","conditioning_set":list(condition),
                "local_attention_relative_error":measurement["relative_error"],"comparison_reference":"conditioned_key_set",
                "actual_retained_tokens":window.k_post.shape[1]-len(ia)-len(condition),"metric_scope":"attention_local"})
    _diversity_merge(window,candidates,store,config)


def _diversity_merge(window,candidates,store,config):
    if len(candidates)<2:return
    size=max({len(indices) for _,indices in candidates},key=lambda n:sum(len(ids)==n for _,ids in candidates))
    ids=[i for i,(_,indices) in enumerate(candidates) if len(indices)==size]
    count=max(1,len(ids)//2);representatives=[ids[0]]
    def distance(i,j):
        a,b=candidates[i][0],candidates[j][0]
        value=(cosine(a.k_pre,b.k_pre)+cosine(a.v,b.v))/2
        return 1-value if np.isfinite(value) else 0.
    while len(representatives)<count:
        representatives.append(max((i for i in ids if i not in representatives),key=lambda i:min(distance(i,j) for j in representatives)))
    assigned={i:[i] for i in representatives}
    for i in ids:
        if i in representatives:continue
        allowed=[j for j in representatives if abs(candidates[i][0].frame-candidates[j][0].frame)<=config.sampling.max_pair_frame_distance]
        if allowed:assigned[min(allowed,key=lambda j:distance(i,j))].append(i)
    groups=[];removed=[]
    for representative,members in assigned.items():
        if len(members)<2:continue
        groups.extend(tuple(candidates[i][1][token] for i in members) for token in range(size))
        removed.extend(token for i in members[1:] for token in candidates[i][1])
    if not groups:return
    for strategy,variant in (("diversity_based_group_merge",ops.merge(window,groups,multiplicity=True)),("diversity_matched_drop",ops.drop(window,removed))):
        measured=evaluate_variant(window,variant,config)
        store.write("E3",{**window.context.row(),"strategy":strategy,"local_attention_relative_error":measured["relative_error"],
            "actual_retained_tokens":variant.token_count,"actual_retention":variant.token_count/window.k_post.shape[1],
            "budget_cohort":f"tokens:{variant.token_count}","grouping_protocol":"farthest_point_joint_KV_representatives_then_nearest_assignment",
            "metric_scope":"attention_local"})


def quantized_variant(window,indices,k_bits,v_bits,group_size):
    initial=ops.identity(window);index=torch.as_tensor(indices,device=window.k_post.device)
    kq=fake_quantize(window.k_post.index_select(1,index),k_bits,group_size)
    vq=fake_quantize(window.v.index_select(1,index),v_bits,group_size)
    k,v=window.k_post.clone(),window.v.clone();k[:,index]=kq.value;v[:,index]=vq.value
    native=(window.k_post.shape[1]-len(indices))*window.k_post.shape[0]*window.k_post.shape[2]*window.k_post.shape[3]*window.k_post.element_size()*2
    costs={"payload_bytes":kq.payload_bytes+vq.payload_bytes+native,"scale_bytes":kq.scale_bytes+vq.scale_bytes,
           "metadata_bytes":kq.metadata_bytes+vq.metadata_bytes,"logical_bytes":kq.logical_bytes+vq.logical_bytes+native,
           "physical_kv_bytes":k.numel()*k.element_size()+v.numel()*v.element_size(),"quantizer":"independent_symmetric_fake_quant_v1",
           "precision_scope":"sampled_candidate_tokens; other/protected tokens retain native dtype"}
    variant=ops.KVVariant(k,v,initial.coordinates,initial.log_count,initial.provenance,{"intervention":"independent_fake_quant"})
    return variant,costs,kq,vq


def _asymmetry(window,blocks,store,config):
    base=window.context.row();indices=sorted({index for block in blocks for index in block})
    for strength in (0.,*config.intervention.strengths):
        for role in ("k","v","kv"):
            variant=ops.perturb(window,indices,kind="gaussian",role=role,target_nmse=strength**2,
                                seed=stable_seed(window.context.identity,"E4",strength))
            measured=evaluate_variant(window,variant,config)
            details=dict(variant.details);details["noise_seed"]=details.pop("seed",None)
            store.write("E4",{**base,**details,"strategy":"matched_relative_kv_rmse","role":role,
                "requested_strength":strength,"strength":strength,"calibration_protocol":"relative_kv_rmse",
                "local_attention_relative_error":measured["relative_error"],"metric_scope":"attention_local",
                "nmse_scope":"intervened_history_tokens","actual_retained_tokens":window.k_post.shape[1]})
    bits=tuple(dict.fromkeys((*config.compression.bits,16)));cache={};options={"k":[],"v":[]}
    for kb in bits:
        for vb in bits:
            variant,costs,kq,vq=quantized_variant(window,indices,kb,vb,config.compression.group_size)
            measurement=evaluate_variant(window,variant,config);error=measurement["relative_error"]
            cache[kb,vb]=(error,costs)
            store.write("E4",{**base,**costs,"strategy":"fake_quant_grid","k_bits":kb,"v_bits":vb,
                "local_attention_relative_error":error,"nmse_k":nmse(window.k_post[:,indices],variant.k_post[:,indices]),
                "nmse_v":nmse(window.v[:,indices],variant.v[:,indices]),"actual_retained_tokens":window.k_post.shape[1],"metric_scope":"attention_local"})
            if vb==16:options["k"].append(PrecisionOption(str(kb),kq.logical_bytes,error,{"bits":kb,"role":"k"}))
            if kb==16:options["v"].append(PrecisionOption(str(vb),vq.logical_bytes,error,{"bits":vb,"role":"v"}))
    def actual(assignment):return cache[int(assignment["k"].name),int(assignment["v"].name)][0]
    frontier=allocation_frontier(options,max_states=config.compression.allocation_max_states,objective=actual)
    for row in frontier["rows"]:
        assignment=row["assignment"];kb,vb=int(assignment["k"].name),int(assignment["v"].name)
        store.write("E4",{**base,**cache[kb,vb][1],"strategy":"joint_attention_precision_oracle","k_bits":kb,"v_bits":vb,
            "local_attention_relative_error":row["error"],"allocation_objective":row["objective_kind"],
            "budget_cohort":f"bytes:{row['total_bytes']}","allocated_candidate_bytes":row["total_bytes"],"metric_scope":"attention_local"})
    for budget in sorted({row["total_bytes"] for row in frontier["rows"]}):
        proxy=allocate(options,budget,max_states=config.compression.allocation_max_states)
        kb,vb=int(proxy["assignment"]["k"].name),int(proxy["assignment"]["v"].name)
        store.write("E4",{**base,**cache[kb,vb][1],"strategy":"sensitivity_based_additive_proxy","k_bits":kb,"v_bits":vb,
            "local_attention_relative_error":cache[kb,vb][0],"proxy_score":proxy["error"],"allocation_objective":proxy["objective_kind"],
            "allocated_candidate_bytes":proxy["total_bytes"],"budget_cohort":f"budget_bytes:{budget}","metric_scope":"attention_local"})
        controls=precision_policy_rows(options,budget,objective=actual,max_states=config.compression.allocation_max_states)
        for row in controls["rows"]:
            kb,vb=int(row["assignment"]["k"].name),int(row["assignment"]["v"].name)
            store.write("E4",{**base,"strategy":row.get("policy",row.get("strategy","precision_control")),
                "k_bits":kb,"v_bits":vb,**cache[kb,vb][1],
                "local_attention_relative_error":row["error"],"allocated_candidate_bytes":row["total_bytes"],
                "budget_cohort":f"budget_bytes:{budget}","metric_scope":"attention_local"})
