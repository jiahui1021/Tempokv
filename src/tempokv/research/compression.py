"""M3 selection and strictly frozen quantization/pruning comparisons."""
from __future__ import annotations
import time
import torch
from .attention_oracle import drop_window,reference
from .capture import stable_seed
from .metrics import relative_error
from .selection import select_history
from .interaction import quant_prune_2x2
from .scores import candidate_scores
from tempokv.spectral.sensitivity import Block,attention_mass


def run_compression(window,candidates,information,store,config):
    blocks=[indices for _,indices in candidates];base=window.context.row()
    baseline=reference(window,config.oracle.query_batch_size)
    if not blocks:
        for exp in set(config.experiments)&{"S3","E7"}:
            store.write(exp,{**base,"strategy":"null","local_attention_relative_error":0.,"actual_retained_tokens":window.k_post.shape[1],"actual_retention":1.})
            store.write(exp,{**base,"status":"not_measured","reason":"no_eligible_clean_history_blocks"})
        return
    errors=[row["local_attention_relative_error"] for row in information["oracle"]]
    if "S3" in config.experiments:
        for retention in (1.,*config.compression.retention):
            for policy in config.compression.policies:
                began=time.perf_counter()
                try:selected=select_history(window,blocks,policy=policy,retention=retention,scores=information["scores"],oracle_errors=errors,
                                             seed=stable_seed(window.context.identity,"S3_random"),online=False)
                except ValueError as exc:
                    if "finite" in str(exc) or "missing" in str(exc):
                        store.write("S3",{**base,"strategy":policy,"retention":retention,"status":"not_measured","reason":str(exc)});continue
                    raise
                scored=(time.perf_counter()-began)*1000
                t=time.perf_counter();changed=drop_window(window,selected["removed_indices"]);gather=(time.perf_counter()-t)*1000
                t=time.perf_counter();actual=reference(changed,config.oracle.query_batch_size);attention=(time.perf_counter()-t)*1000
                payload=(changed.k_post.numel()+changed.v.numel())*changed.k_post.element_size()
                index_bytes=0 if not selected["removed_indices"] else selected["actual_retained_tokens"]*8
                store.write("S3",{**base,**selected,"strategy":policy,"retention":retention,
                    "local_attention_relative_error":relative_error(baseline,actual),"payload_bytes":payload,"metadata_bytes":index_bytes,
                    "logical_bytes":payload+index_bytes,"physical_kv_bytes":(window.k_post.untyped_storage().nbytes()+window.v.untyped_storage().nbytes()),
                    "scoring_ms":scored,"gather_ms":gather,"attention_ms":attention,"timing_scope":"analysis_host_elapsed",
                    "budget_cohort":f"candidate_tokens:{selected['kept_candidate_tokens']}","metric_scope":"attention_local",
                    "storage_contract":"compute-only simulated removal; original backing retained","oracle_scope":"sampled_LOO_sorting_diagnostic_not_global_upper_bound" if policy=="oracle" else None})
    if "E7" in config.experiments:
        if config.source!="bf16":
            store.write("E7",{**base,"status":"not_measured","reason":"full2x2_requires_matched_frozen_BF16_reference; decoded_TempoKV_trajectory_alone_is_insufficient"})
            return
        # The primary 2x2 uses one BF16-derived mask, then freezes it for B and D.
        policy="attention";pairs=tuple(dict.fromkeys(((4,4),*config.compression.mixed_bits)))
        for retention in (1.,*config.compression.retention):
            selected=select_history(window,blocks,policy=policy,retention=retention,scores=information["scores"],oracle_errors=errors,online=False)
            def selection_fn(candidate):
                mass=attention_mass(candidate.q_post,candidate.k_post,[Block(entry.frame,entry.block_id,indices) for entry,indices in candidates])
                scores,_=candidate_scores(candidate,candidates,[{"attention_mass":value} for value in mass])
                selection=select_history(candidate,blocks,policy=policy,retention=retention,scores=scores,oracle_errors=errors,online=False)
                token_scores=torch.full((candidate.k_post.shape[1],),float("nan"))
                for block,score in zip(blocks,scores["attention"]):token_scores[list(block)]=float(score)
                return {**selection,"scores":token_scores}
            for kb,vb in pairs:
                result=quant_prune_2x2(window,kept_indices=selected["kept_indices"],k_bits=kb,v_bits=vb,
                    group_size=config.compression.group_size,query_batch_size=config.oracle.query_batch_size,selection_fn=selection_fn)
                for row in result["rows"]:
                    store.write("E7",{**base,**row,"retention":retention,"k_bits":kb,"v_bits":vb,
                        "interaction":result["interaction"],"D_minus_C":result["D_minus_C"],"B_minus_A":result["B_minus_A"],
                        "selection_consistency":result.get("selection_consistency"),"error_correlation":result.get("error_correlation"),
                        "score_definition":policy,"mask_protocol":"frozen_BF16_mask_for_B_and_D","metric_scope":"attention_local",
                        "interpretation":"nonlinear_relative_error_difference_not_additive_physical_synergy","jensen_correction":False})
