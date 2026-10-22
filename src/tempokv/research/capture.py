"""Bounded clean-block reservoir and continuous fixed-position temporal samples."""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from .transforms import spatial_blocks, spatial_energy, temporal_energy, make_projection, compress_grid, reconstruct_grid
from .metrics import nmse


def stable_seed(*parts): return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=4).digest(), "little")

@dataclass(frozen=True)
class CleanBlock:
    frame: int
    block_id: int
    spatial_indices: tuple[int, ...]
    k_pre: torch.Tensor
    v: torch.Tensor
    freq_k_high: float
    freq_v_high: float
    similarity: float
    priority: int

    @property
    def bytes(self): return (self.k_pre.numel()*self.k_pre.element_size()+self.v.numel()*self.v.element_size())


class CleanBank:
    def __init__(self, config):
        self.config = config; self.blocks = defaultdict(dict); self.trajectories = defaultdict(deque)
        self.bytes = 0; self.dropped = 0; self.rows = 0; self.stream_grids = {}

    @property
    def capture_bytes(self):
        return self.bytes+sum(value.numel()*value.element_size() for history in self.trajectories.values() for _,value in history)

    def add(self, *, sample, layer, head, frame, k, v, prompt, seed):
        """k/v full [H,W,D] for this clean head, copied and freed after statistics."""
        c = self.config; h, w, d = k.shape
        if k.shape != v.shape or frame >= c.sampling.max_clean_frames: return []
        stream = (sample, layer, head); records = []
        if stream in self.stream_grids and self.stream_grids[stream] != (h,w):
            self.bytes-=sum(entry.bytes for entry in self.blocks[stream].values());self.blocks[stream].clear()
            for role in ("k","v"):self.trajectories[(*stream,role)].clear()
        self.stream_grids[stream]=(h,w)
        for role, value in (("k", k), ("v", v)):
            ratios, total, zero = spatial_energy(value[None, None, :, :, None, :], c.spectral.cutoffs, c.eps)
            for cutoff, ratio in ratios.items():
                records.append({"frame": frame, "role": role, "score_type": "spatial_low_energy", "score": float(ratio[0,0,0]),
                    "cutoff": cutoff, "total_energy": float(total[0,0,0]), "near_zero_energy": bool(zero[0,0,0]),
                    "metric_scope": "clean_dense_spatial", "coordinate": "pre_rope" if role == "k" else "raw"})
            target=((h+1)//2,(w+1)//2)
            pooled=compress_grid(value[None,:,:,None,:],make_projection("avg",h,target[0]),make_projection("avg",w,target[1]))
            rebuilt=reconstruct_grid(pooled,(h,w),kind="avg")
            records.append({"frame":frame,"role":role,"score_type":"uniform_pooling_nmse","score":nmse(value,rebuilt[0,:,:,0]),
                "metric_scope":"clean_spatial_pooling_control","pooling_grid":list(target),"coordinate":"pre_rope" if role=="k" else "raw"})
            indices = np.linspace(0,h*w-1,min(h*w,c.spectral.temporal_positions)).round().astype(int)
            key = (*stream, role); history = self.trajectories[key]
            if history and (history[-1][0] != frame-1 or history[-1][1].shape != (len(indices),1,d)): history.clear()
            history.append((frame, value.reshape(h*w,d)[indices, None].float().clone()))
            while len(history) > max(c.spectral.temporal_windows): history.popleft()
            while history and self.capture_bytes > c.sampling.max_capture_mb*1024**2:
                history.popleft();self.dropped+=1
            for length in c.spectral.temporal_windows:
                if len(history) < length: continue
                values = torch.stack([x[1] for x in list(history)[-length:]])
                order = np.random.default_rng(stable_seed(seed,prompt,*stream,role,frame,length)).permutation(length)
                for control, series in (("ordered",values),("shuffled",values[order])):
                    _, power, ratios_t, zero_t = temporal_energy(series,c.spectral.cutoffs,c.eps)
                    for ci,cutoff in enumerate(c.spectral.cutoffs):
                        records.append({"frame":frame,"role":role,"score_type":"temporal_low_energy","score":float(ratios_t[ci,0]),
                            "cutoff":cutoff,"temporal_window":length,"temporal_control":control,"spatial_indices":indices.tolist(),
                            "metric_scope":"clean_contiguous_temporal","coordinate":"pre_rope" if role=="k" else "raw",
                            "near_zero_energy":bool(zero_t[0])})
        for bid, rows, cols, indices in spatial_blocks(h,w,c.sampling.spatial_block_hw):
            kk,vv = k[rows,cols].contiguous(),v[rows,cols].contiguous()
            scores=[]
            for x in (kk,vv):
                ratios,_,_=spatial_energy(x[None,None,:,:,None,:],(c.spectral.score_cutoff,),c.eps)
                scores.append(float(1-ratios[c.spectral.score_cutoff][0,0,0]))
            previous=self.blocks[stream].get((frame-1,bid)); similarity=float("nan")
            if previous and previous.k_pre.numel()==kk.numel():
                values=[]
                for old,new in ((previous.k_pre,kk),(previous.v,vv)):
                    a,b=old.flatten().float(),new.flatten().float();den=a.norm()*b.norm()
                    values.append(float((a@b)/den) if den>c.eps else float("nan"))
                similarity=sum(values)/2
            entry=CleanBlock(frame,bid,indices,kk.reshape(1,-1,1,d).clone(),vv.reshape(1,-1,1,d).clone(),*scores,similarity,
                             stable_seed(seed,prompt,*stream,frame,bid))
            bank=self.blocks[stream]
            if len(bank)>=c.sampling.clean_blocks_per_stream:
                worst=max(bank,key=lambda key:bank[key].priority)
                if bank[worst].priority<=entry.priority: continue
                self.bytes-=bank.pop(worst).bytes
            if self.capture_bytes+entry.bytes > c.sampling.max_capture_mb*1024**2:
                self.dropped+=1;continue
            bank[frame,bid]=entry;self.bytes+=entry.bytes
        return records

    def candidates(self, context):
        stream=(context.sample_id,context.layer_id,context.head_id);bank=self.blocks.get(stream,{})
        if context.key_abs_positions.numel()==0 or self.stream_grids.get(stream)!=(context.grid_h,context.grid_w):return []
        lookup={int(p):i for i,p in enumerate(context.key_abs_positions.tolist())}
        result=[]
        for key,entry in list(bank.items()):
            absolute=tuple(entry.frame*context.frame_tokens+i for i in entry.spatial_indices)
            if max(absolute)<int(context.key_abs_positions[0]) and entry.frame>=context.sink_frames:
                self.bytes-=bank.pop(key).bytes;continue
            if not all(p in lookup for p in absolute):continue
            indices=tuple(lookup[p] for p in absolute)
            if any(context.protected_mask[i] for i in indices):continue
            result.append((entry,indices))
        result.sort(key=lambda pair:pair[0].priority)
        return result[:self.config.sampling.max_history_blocks_per_event]
