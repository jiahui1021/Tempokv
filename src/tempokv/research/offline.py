"""Rebuild local diagnostics from an explicit full-window frozen event.

No inverse RoPE is used to invent pre-K. A partial clean reservoir remains a
partial candidate set; the complete actual post-RoPE K/V window is retained.
"""
from pathlib import Path
import torch
from .capture import CleanBank,stable_seed
from .compression import run_compression
from .diagnostics import write_local
from .frozen_io import load_frozen
from .io import RunStore


def run_offline(config,path):
    if set(config.experiments)&{'E2','E5','E6'} or config.intervention.paired_rollout:
        raise ValueError('frozen input supports local experiments only; rollout requires its own causal replay')
    window,metadata=load_frozen(path,max_capture_mb=config.sampling.max_capture_mb)
    if window.q_post.shape[2]!=1:raise ValueError('local scoring requires one selected head per frozen event')
    bank=CleanBank(config);ctx=window.context
    records=[]
    if window.k_pre is not None:
        positions=ctx.key_abs_positions.tolist();lookup={p:i for i,p in enumerate(positions)}
        valid=window.pre_valid_mask if window.pre_valid_mask is not None else torch.ones(len(positions),dtype=torch.bool)
        for frame in sorted({p//ctx.frame_tokens for p in positions if p<ctx.current_chunk_start}):
            ids=[lookup.get(frame*ctx.frame_tokens+i,-1) for i in range(ctx.frame_tokens)]
            # Dense spatial FFT is valid only for a completely captured clean frame.
            if min(ids)<0 or not bool(valid[ids].all()):continue
            k=window.k_pre[0,ids,0].reshape(ctx.grid_h,ctx.grid_w,-1).cpu()
            v=window.v[0,ids,0].reshape_as(k).cpu()
            records.extend(bank.add(sample=ctx.sample_id,layer=ctx.layer_id,head=ctx.head_id,frame=frame,k=k,v=v,prompt=ctx.prompt_index,seed=ctx.seed))
    candidates=bank.candidates(ctx)
    # Sparse exact pre-K from a live reservoir can cover individual blocks even
    # when no entire frame was captured. Rebuild those blocks without FFTing gaps.
    if window.k_pre is not None:
        from .capture import CleanBlock
        from .transforms import spatial_blocks,spatial_energy
        found={(entry.frame,entry.block_id) for entry,_ in candidates}
        lookup={int(p):i for i,p in enumerate(ctx.key_abs_positions.tolist())}
        valid=window.pre_valid_mask if window.pre_valid_mask is not None else torch.ones(window.k_post.shape[1],dtype=torch.bool)
        for frame in sorted({p//ctx.frame_tokens for p in lookup if p<ctx.current_chunk_start}):
            for bid,rows,cols,spatial in spatial_blocks(ctx.grid_h,ctx.grid_w,config.sampling.spatial_block_hw):
                ids=tuple(lookup.get(frame*ctx.frame_tokens+p,-1) for p in spatial)
                if (frame,bid) in found or min(ids)<0 or any(ctx.protected_mask[i] or not valid[i] for i in ids):continue
                kk=window.k_pre[:,ids].cpu();vv=window.v[:,ids].cpu();freq=[]
                for value in (kk,vv):
                    dense=value.reshape(1,1,len(range(*rows.indices(ctx.grid_h))),len(range(*cols.indices(ctx.grid_w))),1,-1)
                    ratios,_,_=spatial_energy(dense,(config.spectral.score_cutoff,),config.eps)
                    freq.append(float(1-ratios[config.spectral.score_cutoff][0,0,0]))
                entry=CleanBlock(frame,bid,spatial,kk,vv,*freq,float('nan'),stable_seed(ctx.identity,frame,bid))
                candidates.append((entry,ids))
    candidates.sort(key=lambda pair:pair[0].priority);candidates=candidates[:config.sampling.max_history_blocks_per_event]
    original_origin=str(metadata.get('metadata',{}).get('data_origin','unknown'))
    origin=('synthetic_offline' if original_origin.startswith('synthetic') else
            'offline_frozen_real' if original_origin in {'runtime_bf16','runtime_tempokv'} else 'unknown')
    # Unknown snapshots must never be promoted to real model evidence.
    store=RunStore(config,origin=origin,seed=ctx.seed,prompt_index=ctx.prompt_index,
                   metadata={'snapshot':str(Path(path).resolve()),'frozen_header':metadata,'offline_backend':'cpu_sdpa_preserved_dtype'})
    if 'S1' in config.experiments:
        if not records:store.write('S1',{**ctx.row(),'status':'not_measured','reason':'no_complete_exact_pre_RoPE_clean_frame'})
        for row in records:store.write('S1',{**ctx.row(),**row})
    information=write_local(window,candidates,store,config)
    run_compression(window,candidates,information,store,config)
    return store.close(extra={'frozen_input':str(Path(path).resolve()),'reference_backend':'cpu_sdpa_preserved_dtype','real_generation':'not_run'})
